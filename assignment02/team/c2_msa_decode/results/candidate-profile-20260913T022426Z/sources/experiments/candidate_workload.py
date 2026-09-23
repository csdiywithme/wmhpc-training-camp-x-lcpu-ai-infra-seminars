"""Post-profile candidate exploration. All configurations and failures are saved."""
import argparse
import hashlib
import itertools
import json
import random
from pathlib import Path
import sys
import traceback

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import candidate
import baseline_workload as base


def prepare(case, config):
    output = torch.empty_like(case["q"])
    rq, hq, dim = case["q"].shape
    total_tiles = case["topk"] * (128 // config["token_tile"])
    target = max(1, min(total_tiles, 256 // max(1, rq * case["num_kv_heads"])))
    splits = config.get("splits") or 1 << (target.bit_length() - 1)
    workspace = (torch.empty(splits, rq, hq, dim, device="cuda", dtype=case["q"].dtype),
                 torch.empty(splits, rq, hq, device="cuda", dtype=torch.float32))
    def fn():
        candidate.decode(case["q"], case["kv_cache"], case["topk_idx"], case["block_table"],
                         case["seq_lens"], case["num_kv_heads"], case["sm_scale"], output,
                         case["decode_query_len"], case.get("k_scale"), case.get("v_scale"),
                         workspace=workspace, **config)
    names = ("_page_decode_kernel" if config.get("original_partial") else "_subpage_decode_kernel",
             "_merge_feature_tiles" if config.get("merge_tile") else "_merge_topk_attn_out_kernel")
    originals = [getattr(candidate, name) for name in names]
    captures = [base.LaunchCapture(kernel) for kernel in originals]
    try:
        for name, cap in zip(names, captures):
            setattr(candidate, name, cap)
        fn()
    finally:
        for name, kernel in zip(names, originals):
            setattr(candidate, name, kernel)
    calls = [(kernel, *cap.calls[-1]) for kernel, cap in zip(originals, captures)]
    return fn, output, calls, splits


def configs():
    for tile, stages in itertools.product((128, 64, 32), (1, 2, 3)):
        yield dict(token_tile=tile, num_warps=4, num_stages=stages)
    for tile in (128, 64, 32):
        yield dict(token_tile=tile, num_warps=8, num_stages=1)
    for splits in (16, 32, 64):
        yield dict(token_tile=64, num_warps=4, num_stages=1, splits=splits)


def tune(output, batch=None, tp=None, dtype=None):
    rows = []
    shapes = [(batch, tp, dtype)] if batch is not None else [
        (b, t, d) for t in (1, 4) for b in (1, 4, 8, 16) for d in ("bf16", "fp8")]
    for b, t, d in shapes:
        case = base.inputs(b, t, d)
        baseline_out, baseline_calls = base.capture(case)
        partial, merge = [base.launcher(call) for call in baseline_calls]
        def baseline_fn():
            partial(); merge()
        gold = base.reference(case)
        baseline_time = base.graph_time(baseline_fn, calls=64)
        for index, config in enumerate(configs()):
            row = dict(batch=b, tp=t, dtype=d, config=config, baseline=baseline_time)
            try:
                fn, out, calls, splits = prepare(case, config)
                torch.cuda.synchronize()
                delta = out.double() - gold
                err = float(delta.norm() / gold.norm().clamp_min(1e-30))
                row.update(finite=bool(torch.isfinite(out).all()), nrmse=err,
                           max_abs=float(delta.abs().max()), splits=splits)
                # Exploration only. Final acceptance uses the separately frozen
                # full suite, never this convenient preliminary scalar gate.
                if not row["finite"] or err > .02:
                    row["status"] = "CORRECTNESS_FAILED"
                else:
                    compiled = calls[0][-1]
                    row.update(status="MEASURED", candidate=base.graph_time(fn, calls=64),
                               registers=compiled.n_regs, spills=compiled.n_spills,
                               shared_bytes=compiled.metadata.shared)
                    row["speedup"] = baseline_time["median_us"] / row["candidate"]["median_us"]
            except Exception as error:
                row.update(status="ERROR", error=str(error), traceback=traceback.format_exc(limit=5))
            rows.append(row)
            base.dump(output / "tuning.json", rows)
            print(json.dumps({k: v for k, v in row.items() if k not in ("baseline", "candidate", "traceback")}), flush=True)
    return rows


def tune_merge(output):
    rows = []
    for tp, batch, dtype in itertools.product((1, 4), (1, 4, 8, 16), ("bf16", "fp8")):
        case = base.inputs(batch, tp, dtype)
        original_out, calls = base.capture(case)
        partial, merge = [base.launcher(call) for call in calls]
        reference = base.reference(case)
        def baseline_fn():
            partial(); merge()
        baseline_time = base.graph_time(baseline_fn, calls=64)
        args = calls[1][2]
        parts, logs = args[0], args[1]
        output_tensor = torch.empty_like(original_out)
        for tile, warps in itertools.product((16, 32, 64, 128), (1, 2, 4)):
            row = dict(batch=batch, tp=tp, dtype=dtype, tile=tile, warps=warps,
                       baseline=baseline_time, partial="untouched upstream")
            try:
                def merge_fn():
                    return candidate.merge_features(parts, logs, output_tensor, tile, warps)
                compiled = merge_fn()
                def chain():
                    partial(); merge_fn()
                chain(); torch.cuda.synchronize()
                delta = output_tensor.double() - reference
                row.update(finite=bool(torch.isfinite(output_tensor).all()),
                           nrmse=float(delta.norm() / reference.norm().clamp_min(1e-30)),
                           max_abs=float(delta.abs().max()))
                if not row["finite"] or row["nrmse"] > .02:
                    row["status"] = "CORRECTNESS_FAILED"
                else:
                    row.update(status="MEASURED", candidate=base.graph_time(chain, calls=64),
                               merge=base.graph_time(merge_fn, calls=64),
                               registers=compiled.n_regs, spills=compiled.n_spills,
                               shared_bytes=compiled.metadata.shared)
                    row["speedup"] = baseline_time["median_us"] / row["candidate"]["median_us"]
            except Exception as error:
                row.update(status="ERROR", error=str(error), traceback=traceback.format_exc(limit=5))
            rows.append(row)
            base.dump(output / "merge-tuning.json", rows)
            print(json.dumps({k:v for k,v in row.items() if k not in ("baseline","candidate","merge","traceback")}), flush=True)
    return rows


def capture_graph(fn, calls=64):
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


def paired(output):
    rows = []
    random_order = random.Random(20260913)
    for tp, batch, dtype, seed, pdl in itertools.product(
            (1, 4), (1, 4, 8, 16), ("bf16", "fp8"), (101, 307), (False, True)):
        # Inputs come from independent performance seeds, not the tuning seed0.
        case = base.inputs(batch, tp, dtype, pdl, seed=seed)
        old_out, old_calls = base.capture(case)
        partial, merge = [base.launcher(c) for c in old_calls]
        def original():
            partial(); merge()
        cfg = candidate.selected_config(case, use_pdl=pdl)
        improved, new_out, new_calls, splits = prepare(case, cfg)
        gold = base.reference(case)
        improved();torch.cuda.synchronize()
        correctness = {}
        for name, value in (("baseline",old_out),("candidate",new_out)):
            delta=value.double()-gold
            correctness[name]=dict(finite=bool(torch.isfinite(value).all()),
                nrmse=float(delta.norm()/gold.norm().clamp_min(1e-30)),max_abs=float(delta.abs().max()))
            assert correctness[name]["finite"] and correctness[name]["nrmse"]<.02
        graphs={"baseline":capture_graph(original),"candidate":capture_graph(improved)}
        samples={name:[] for name in graphs};orders=[]
        for repeat in range(21):
            names=list(graphs);random_order.shuffle(names);orders.append(names)
            for name in names:
                a,b=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
                a.record();graphs[name].replay();b.record();b.synchronize()
                samples[name].append(a.elapsed_time(b)*1000/64)
        row=dict(tp=tp,batch=batch,dtype=dtype,seed=seed,pdl=pdl,config=cfg,splits=splits,
                 correctness=correctness,orders=orders,graph_calls=64,repeats=21,
                 baseline=base.stats(samples["baseline"]),candidate=base.stats(samples["candidate"]))
        row["paired_speedup_samples"]=[a/b for a,b in zip(samples["baseline"],samples["candidate"])]
        row["paired_speedup"]=base.stats(row["paired_speedup_samples"])
        row["speedup"]=row["baseline"]["median_us"]/row["candidate"]["median_us"]
        directory=output/f"tp{tp}-b{batch}-{dtype}-seed{seed}-pdl{int(pdl)}"
        # One sample per shape/PDL is sufficient to identify the actual kernel.
        if seed==101:
            base.compiled_artifacts(new_calls,directory)
        rows.append(row);base.dump(output/"paired.json",rows)
        print(json.dumps({k:row[k] for k in ("tp","batch","dtype","seed","pdl","speedup","config")}),flush=True)
    return rows


def profile_selected(output):
    # Four requested launches: original partial + selected merge, B1 then B16.
    for batch in (1,16):
        case=base.inputs(batch,1,"bf16")
        fn,out,calls,splits=prepare(case,candidate.selected_config(case))
        torch.cuda.synchronize()
        base.compiled_artifacts(calls,output/f"tp1-b{batch}-bf16")
    print("SELECTED_PROFILE_DONE",flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--mode", choices=("tune", "merge", "paired", "profile"), default="tune")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    base.dump(args.output / "candidate-environment.json", dict(
        gpu=str(torch.cuda.get_device_properties(0)), torch=torch.__version__,
        candidate_sha256=hashlib.sha256((ROOT / "candidate.py").read_bytes()).hexdigest(),
        scope="exploration; fixed seed0 hot-address graph; final paired heldout benchmark still required"))
    if args.mode == "profile":
        profile_selected(args.output)
    elif args.mode == "paired":
        paired(args.output)
    elif args.mode == "merge":
        tune_merge(args.output)
    else:
        tune(args.output, args.batch, args.tp, args.dtype)


if __name__ == "__main__":
    main()
