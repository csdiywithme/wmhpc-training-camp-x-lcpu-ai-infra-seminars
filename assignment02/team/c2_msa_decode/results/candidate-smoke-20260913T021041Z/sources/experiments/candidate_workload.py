"""Post-profile candidate exploration. All configurations and failures are saved."""
import argparse
import hashlib
import itertools
import json
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
    names = ("_subpage_decode_kernel", "_merge_topk_attn_out_kernel")
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--dtype", default="bf16")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    base.dump(args.output / "candidate-environment.json", dict(
        gpu=str(torch.cuda.get_device_properties(0)), torch=torch.__version__,
        candidate_sha256=hashlib.sha256((ROOT / "candidate.py").read_bytes()).hexdigest(),
        scope="exploration; fixed seed0 hot-address graph; final paired heldout benchmark still required"))
    tune(args.output, args.batch, args.tp, args.dtype)


if __name__ == "__main__":
    main()
