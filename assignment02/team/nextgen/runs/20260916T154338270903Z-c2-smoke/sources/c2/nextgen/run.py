"""Bounded complete-attention smoke/acceptance/paired-performance experiments.

Run from /opt/c2, after CPU compilation. No JIT compilation is performed here.
"""
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def check(case, value, gold, bounds=None):
    from validation import suite
    from validation.run_validation import failures_for
    metrics = suite.evaluate(case, value, gold)
    failures = failures_for(metrics, bounds)
    if failures:
        raise AssertionError(json.dumps({"metrics": metrics, "failures": failures}))
    return metrics


def frozen_manifest(path):
    from validation import suite
    manifest = json.loads(path.read_text())
    if manifest.get("status") != "CALIBRATION_FROZEN":
        raise ValueError("Requires the successful existing frozen manifest")
    protocol = suite.protocol(manifest["protocol"]["tier"])
    if suite.protocol_digest(protocol) != manifest["protocol_digest"]:
        raise ValueError("Frozen protocol digest mismatch")
    return manifest


def capture(fn, calls):
    import torch
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(calls):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    return graph


def smoke(args):
    import torch
    from nextgen.adapter import prepare
    from validation import suite
    specs = [
        suite.CaseSpec("single_page_bf16", 1, (127,)),
        suite.CaseSpec("multiple_pages_bf16", 4, (2305,)),
        suite.CaseSpec("causal_dql_bf16", 1, (129, 257), 4, pattern="causal_probe"),
        suite.CaseSpec("fp8_scalar_padding", 1, (0, 129, 2305), storage="fp8_scalar"),
        suite.CaseSpec("fp8_token_strides", 4, (129, 2305), storage="fp8_token"),
    ]
    rows = []
    for spec in specs:
        case = suite.make_case(spec, 101)
        gold = suite.gold(case)
        baseline_metrics = check(case, suite.baseline(case), gold)
        for splits in (1, 4, 16):
            item = {"case": spec.case_id, "splits": splits, "baseline": baseline_metrics}
            try:
                prepared = prepare(case, splits=splits)
                item.update(actual_backend=prepared["actual_backend"], kv_conversion=prepared["kv_conversion"], fallback=False)
                value = prepared["chain"]()
                torch.cuda.synchronize()
                item["eager"] = check(case, value, gold)
                graph = capture(prepared["chain"], 2)
                graph.replay(); torch.cuda.synchronize()
                item["graph"] = check(case, value, gold)
                item["status"] = "PASS"
            except Exception as error:
                item.update(status="FAIL", error=str(error), traceback=traceback.format_exc())
                rows.append(item); save(args.output / "smoke.json", rows)
                raise
            rows.append(item); save(args.output / "smoke.json", rows)
            print(json.dumps(item), flush=True)
    return rows


def verify(args):
    commands = []
    manifest = args.manifest
    os.environ["C2_NEXTGEN_AUDIT"] = str((args.output / "adapter_calls.jsonl").resolve())
    if manifest is not None and not manifest.is_file():
        raise FileNotFoundError(f"Frozen manifest not found: {manifest}; install the existing frozen manifest")
    entry = [sys.executable, "-u", str(ROOT / "validation/run_validation.py")]
    commands.append(entry + ["verify", "--manifest", str(manifest), "--adapter", "nextgen.adapter:run",
                             "--output", str(args.output / "heldout.json")])
    save(args.output / "validation-commands.json", commands)
    for index, command in enumerate(commands):
        with (args.output / f"validation-{index}.log").open("w") as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Validation command {index} failed: {command}; see saved log")


def bench(args):
    import torch
    import baseline_workload as base
    import candidate_workload as old
    import candidate
    from nextgen.adapter import prepare
    from validation import suite
    manifest = frozen_manifest(args.manifest)
    rows, rng = [], random.Random(20260916)
    for tp, batch, storage, seed in itertools.product(args.tps, args.batches, args.storages, args.seeds):
        spec = suite.CaseSpec(f"tp{tp}-b{batch}-{storage}", 4 // tp, (8192,) * batch, storage=storage)
        case = suite.make_case(spec, seed)
        bounds = manifest["thresholds"][spec.family]
        gold = suite.gold(case)
        base.sa.current_platform.is_arch_support_pdl = lambda: False
        base_out, calls = base.capture(case)
        base_part, base_merge = [base.launcher(call) for call in calls]
        def baseline_chain():
            base_part(); base_merge()
        old_fn, old_out, _, _ = old.prepare(case, candidate.selected_config(case, use_pdl=False))
        prepared = prepare(case, splits=args.splits, merge=args.merge)
        prepared["chain"](); torch.cuda.synchronize()
        fns = {"baseline": baseline_chain, "merge_only": old_fn, "tcgen05": prepared["chain"]}
        values = {"baseline": base_out, "merge_only": old_out, "tcgen05": prepared["output"]}
        eager = {name: check(case, value, gold, bounds) for name, value in values.items()}
        graphs = {name: capture(fn, args.calls) for name, fn in fns.items()}
        # Avoid sub-millisecond event intervals. Estimate each complete graph,
        # then use the same replay count for all paths so the fastest estimated
        # path reaches the target duration, subject to the explicit safety cap.
        calibration = {}
        for name, graph in graphs.items():
            calibration[name] = []
            for _ in range(3):
                begin, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                begin.record(); graph.replay(); end.record(); end.synchronize()
                calibration[name].append(begin.elapsed_time(end) * 1000)
        fastest_graph_us = min(statistics.median(values) for values in calibration.values())
        if fastest_graph_us <= 0 or not math.isfinite(fastest_graph_us):
            raise ValueError(f"Invalid graph duration estimate: {calibration}")
        replays = min(args.max_replays, max(1, math.ceil(args.sample_ms * 1000 / fastest_graph_us)))
        denominator = args.calls * replays
        samples = {name: [] for name in fns}; orders = []
        for repeat in range(args.repeats):
            order = list(fns); rng.shuffle(order); orders.append(order)
            for name in order:
                begin, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                begin.record()
                for _ in range(replays):
                    graphs[name].replay()
                end.record(); end.synchronize()
                samples[name].append(begin.elapsed_time(end) * 1000 / denominator)
        after = {name: check(case, value, gold, bounds) for name, value in values.items()}
        medians = {name: statistics.median(v) for name, v in samples.items()}
        row = {"tp": tp, "batch": batch, "storage": storage, "seed": seed, "pdl": False,
               "actual_backend": prepared["actual_backend"], "kv_conversion": prepared["kv_conversion"], "fallback": False,
               "frozen_protocol_digest": manifest["protocol_digest"], "frozen_family_bounds": bounds,
               "splits": prepared["splits"], "merge": args.merge,
               "calls_per_graph": args.calls, "repeats": args.repeats, "orders": orders,
               "replays_per_sample": replays, "invocations_per_sample": denominator,
               "target_sample_ms": args.sample_ms, "replay_cap": args.max_replays,
               "replay_cap_reached": replays == args.max_replays,
               "calibration_graph_us": calibration,
               "estimated_shortest_sample_ms": fastest_graph_us * replays / 1000,
               "samples_us": samples, "median_us": medians, "eager": eager, "post_graph": after,
               "speedup_vs_original": medians["baseline"] / medians["tcgen05"],
               "speedup_vs_merge_only": medians["merge_only"] / medians["tcgen05"]}
        rows.append(row); save(args.output / "paired.json", rows)
        print(json.dumps({k: row[k] for k in ("tp", "batch", "storage", "seed", "median_us", "speedup_vs_original", "speedup_vs_merge_only")}), flush=True)


def profile(args):
    """One NVTX-delimited complete chain for an externally bounded NCU run.

    All input construction, reference calculation, kernel warmup, and numeric
    checks are outside the selected NVTX range. No profiler latency is used as
    a performance benchmark. The root orchestration owns NCU options/timeouts.
    """
    import torch
    import baseline_workload as base
    import candidate_workload as old
    import candidate
    from nextgen.adapter import prepare
    from validation import suite
    if any(len(v) != 1 for v in (args.tps, args.batches, args.storages, args.seeds)):
        raise ValueError("profile requires exactly one --tps, --batches, --storages and --seeds value")
    tp, batch, storage, seed = args.tps[0], args.batches[0], args.storages[0], args.seeds[0]
    manifest = frozen_manifest(args.manifest)
    spec = suite.CaseSpec(f"profile-tp{tp}-b{batch}-{storage}", 4 // tp, (8192,) * batch, storage=storage)
    case = suite.make_case(spec, seed)
    bounds = manifest["thresholds"][spec.family]
    gold = suite.gold(case)
    base.sa.current_platform.is_arch_support_pdl = lambda: False
    base_out, calls = base.capture(case)
    base_part, base_merge = [base.launcher(call) for call in calls]
    def baseline_chain():
        base_part(); base_merge()
    old_fn, old_out, _, _ = old.prepare(case, candidate.selected_config(case, use_pdl=False))
    prepared = prepare(case, splits=args.splits, merge=args.merge)
    fns = {"baseline": baseline_chain, "merge_only": old_fn, "tcgen05": prepared["chain"]}
    values = {"baseline": base_out, "merge_only": old_out, "tcgen05": prepared["output"]}
    for fn in fns.values():
        for _ in range(5):
            fn()
    torch.cuda.synchronize()
    before = {name: check(case, value, gold, bounds) for name, value in values.items()}
    marker = "c2_nextgen_profile_" + args.profile_target
    row = {"status": "READY", "target": args.profile_target, "nvtx_range": marker,
           "tp": tp, "batch": batch, "storage": storage, "seed": seed,
           "splits": prepared["splits"], "merge": args.merge, "pdl": False,
           "actual_backend": prepared["actual_backend"] if args.profile_target == "tcgen05" else
                             "upstream_triton_partial_with_" + ("feature_merge" if args.profile_target == "merge_only" else "original_merge"),
           "fallback": False, "chain_replays": args.profile_replays, "warmups_per_path": 5,
           "kernels_per_chain": 2, "before": before,
           "frozen_protocol_digest": manifest["protocol_digest"], "frozen_family_bounds": bounds,
           "scope": "NVTX-delimited eager complete chain; profiler mechanism evidence only, not hot-Graph benchmark latency"}
    save(args.output / "profile.json", row)
    torch.cuda.nvtx.range_push(marker)
    try:
        for _ in range(args.profile_replays):
            fns[args.profile_target]()
    finally:
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    row.update(status="COMPLETE", after={name: check(case, value, gold, bounds) for name, value in values.items()})
    save(args.output / "profile.json", row)
    print(json.dumps(row), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "verify", "bench", "profile"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--build-dir", type=Path, default=Path("/tmp/nextgen-build"))
    parser.add_argument("--manifest", type=Path, default=Path("/opt/c2/validation/frozen_calibration.json"))
    parser.add_argument("--tier", choices=("quick", "full"), default="full")
    parser.add_argument("--tps", type=lambda s: [int(x) for x in s.split(",")], default=[1, 4])
    parser.add_argument("--batches", type=lambda s: [int(x) for x in s.split(",")], default=[1, 4, 8, 16])
    parser.add_argument("--storages", type=lambda s: s.split(","), default=["bf16", "fp8_scalar"])
    parser.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")], default=[101, 307])
    parser.add_argument("--splits", type=int, choices=(1, 2, 4, 8, 16))
    parser.add_argument("--merge", choices=("feature", "original"), default="feature")
    parser.add_argument("--calls", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--sample-ms", type=float, default=10.0)
    parser.add_argument("--max-replays", type=int, default=64)
    parser.add_argument("--profile-target", choices=("baseline", "merge_only", "tcgen05"), default="tcgen05")
    parser.add_argument("--profile-replays", type=int, default=1)
    args = parser.parse_args()
    if args.calls < 1 or args.repeats < 1 or args.profile_replays < 1 or args.max_replays < 1 or not math.isfinite(args.sample_ms) or args.sample_ms <= 0:
        parser.error("calls/repeats/profile-replays/max-replays/sample-ms must be positive and finite")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "environment.json").exists():
        parser.error("Existing run output; use a fresh directory to preserve failures")
    os.environ["C2_NEXTGEN_BUILD"] = str(args.build_dir.resolve())
    import torch
    import triton
    torch.set_grad_enabled(False)
    info = {"gpu": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
            "torch": torch.__version__, "triton": triton.__version__, "cuda": torch.version.cuda,
            "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "sources": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.iterdir() if p.suffix in (".py", ".cu", ".cpp")},
            "scope": "Single GPU, PDL off, complete partial+merge; BF16 MMA/P and FP32 accumulators; hot fixed-address Graph; no model/service E2E"}
    build_manifest = args.build_dir / "build.json"
    if build_manifest.exists():
        info["build"] = json.loads(build_manifest.read_text())
        for filename, digest in info["build"]["source_hashes"].items():
            if info["sources"].get(filename) != digest:
                raise RuntimeError(f"Stale binary: current {filename} does not match build manifest")
    save(args.output / "environment.json", info)
    try:
        globals()[args.mode](args)
    except Exception as error:
        save(args.output / "failure.json", {"error": str(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
