"""Measure the pinned, unmodified C2 Triton kernels; no candidate optimization.

The capture adapter records existing wrapper launches so individual kernels can
be timed with the exact same arguments and preallocated buffers. Graph timings
and allocating public-wrapper timings are intentionally reported separately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
import triton

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
import vllm_shim
from synth import make_case

sa = vllm_shim.load_sparse_attn()


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")


def quantile(xs, p):
    ordered = sorted(xs)
    x = (len(xs) - 1) * p
    lo = int(x)
    hi = min(lo + 1, len(xs) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (x - lo)


def stats(samples):
    return dict(mean_us=statistics.mean(samples), median_us=statistics.median(samples),
                p05_us=quantile(samples, .05), p95_us=quantile(samples, .95),
                min_us=min(samples), max_us=max(samples), samples_us=samples)


class LaunchCapture:
    def __init__(self, original):
        self.original = original
        self.calls = []

    def __getitem__(self, grid):
        def call(*args, **kwargs):
            compiled = self.original[grid](*args, **kwargs)
            self.calls.append((grid, args, kwargs, compiled))
            return compiled
        return call


def inputs(batch, tp, dtype, pdl=False, seed=0):
    sa.current_platform.is_arch_support_pdl = lambda: pdl
    case = make_case(num_reqs=batch, seq_range=(8192, 8192),
                     num_kv_heads=4 // tp, seed=seed)
    # Preserve production's token-major source buffer, exposed as strided
    # [Hkv, total_q, topk] to Triton; this transpose performs no data copy.
    case["topk_idx"] = case["topk_idx"].transpose(0, 1).contiguous().transpose(0, 1)
    if dtype == "fp8":
        kv = case["kv_cache"]
        k_scale = torch.tensor(.25, device="cuda", dtype=torch.float32)
        v_scale = torch.tensor(.5, device="cuda", dtype=torch.float32)
        quantized = torch.empty_like(kv, dtype=torch.float8_e4m3fn)
        quantized[..., :128] = (kv[..., :128].float() / k_scale).to(quantized.dtype)
        quantized[..., 128:] = (kv[..., 128:].float() / v_scale).to(quantized.dtype)
        case.update(kv_cache=quantized, k_scale=k_scale, v_scale=v_scale)
    return case


def wrapper(case, out):
    return sa.minimax_m3_sparse_attn_decode(
        case["q"], case["kv_cache"], case["topk_idx"], case["block_table"],
        case["seq_lens"], case["num_kv_heads"], case["sm_scale"], out,
        case["decode_query_len"], k_scale=case.get("k_scale"),
        v_scale=case.get("v_scale"))


def capture(case):
    out = torch.empty_like(case["q"])
    names = ("_gqa_sparse_decode_kernel", "_merge_topk_attn_out_kernel")
    originals = [getattr(sa, name) for name in names]
    captures = [LaunchCapture(fn) for fn in originals]
    try:
        for name, cap in zip(names, captures):
            setattr(sa, name, cap)
        wrapper(case, out)
    finally:
        for name, fn in zip(names, originals):
            setattr(sa, name, fn)
    torch.cuda.synchronize()
    return out, [(original, *cap.calls[-1]) for original, cap in zip(originals, captures)]


def launcher(call):
    kernel, grid, args, kwargs, compiled = call
    return lambda: kernel[grid](*args, **kwargs)


def graph_time(fn, *, calls=32, repeats=9):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(calls):
                fn()
    torch.cuda.current_stream().wait_stream(stream)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / calls)
    return dict(**stats(samples), graph_calls=calls, repeats=repeats,
                scope="CUDA-event average per invocation of a repeated CUDA Graph; hot fixed data")


def api_time(case, calls=40, repeats=5):
    for _ in range(5):
        wrapper(case, torch.empty_like(case["q"]))
    torch.cuda.synchronize()
    device, wall = [], []
    for _ in range(repeats):
        start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        begin = time.perf_counter()
        start.record()
        for _ in range(calls):
            wrapper(case, torch.empty_like(case["q"]))
        end.record()
        end.synchronize()
        wall.append((time.perf_counter() - begin) * 1e6 / calls)
        device.append(start.elapsed_time(end) * 1000 / calls)
    return dict(device=stats(device), synchronized_wall=stats(wall), calls=calls,
                scope="allocating output and upstream partial workspace; Python submission included")


def reference(case):
    """Independent FP64 attention over the same quantized/dequantized inputs."""
    q = case["q"].double()
    kv = case["kv_cache"]
    use_fp8 = kv.dtype == torch.float8_e4m3fn
    if use_fp8:
        # Advanced indexing does not support every FP8 dtype/backend. The
        # exactly representable cast mirrors the baseline's first conversion.
        kv = kv.to(case["q"].dtype)
    topk, bt = case["topk_idx"].cpu(), case["block_table"].cpu()
    out = torch.empty_like(q)
    for request in range(q.shape[0]):
        for head in range(case["num_kv_heads"]):
            pages = bt[request, topk[head, request].long()].to("cuda", dtype=torch.long)
            selected = kv[pages, head].reshape(-1, 256)
            k, v = selected[:, :128], selected[:, 128:]
            if use_fp8:
                k = (k.to(case["q"].dtype).float() * case["k_scale"]).to(case["q"].dtype)
                v = (v.to(case["q"].dtype).float() * case["v_scale"]).to(case["q"].dtype)
            qq = q[request, head * 16:(head + 1) * 16]
            pp = torch.softmax(qq @ k.double().T * case["sm_scale"], dim=-1)
            out[request, head * 16:(head + 1) * 16] = pp @ v.double()
    return out


def compiled_artifacts(calls, directory):
    rows = []
    for label, call in zip(("partial", "merge"), calls):
        kernel, grid, args, kwargs, compiled = call
        meta = getattr(compiled, "metadata", None)
        row = dict(label=label, grid=list(grid), metadata=meta._asdict() if hasattr(meta, "_asdict") else str(meta),
                   n_regs=getattr(compiled, "n_regs", None), n_spills=getattr(compiled, "n_spills", None),
                   launch_kwargs=kwargs)
        for ext in ("ptx", "cubin", "ttgir"):
            data = compiled.asm.get(ext)
            if data is not None:
                path = directory / f"{label}.{ext}"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data if isinstance(data, bytes) else data.encode())
                row[f"{ext}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        cubin = directory / f"{label}.cubin"
        if cubin.exists():
            proc = subprocess.run(["cuobjdump", "--dump-sass", str(cubin)], capture_output=True, text=True)
            (directory / f"{label}.sass").write_text(proc.stdout)
            row["sass_returncode"] = proc.returncode
            row["sass_stderr"] = proc.stderr
        rows.append(row)
    dump(directory / "compiled.json", rows)
    return rows


def run_one(batch, tp, dtype, directory, pdl=False, profile=False):
    case = inputs(batch, tp, dtype, pdl)
    out, calls = capture(case)
    if profile:
        return dict(batch=batch, tp=tp, dtype=dtype, pdl=pdl, profile_launches_completed=True)
    compiled = compiled_artifacts(calls, directory)
    gold = reference(case)
    delta = out.double() - gold
    correctness = dict(finite=bool(torch.isfinite(out).all()),
                       max_abs=float(delta.abs().max()),
                       nrmse=float(delta.norm() / gold.norm().clamp_min(1e-30)))
    if not correctness["finite"] or correctness["nrmse"] > .02:
        raise AssertionError(correctness)
    partial, merge = [launcher(call) for call in calls]
    def chain():
        partial()
        merge()
    row = dict(batch=batch, tp=tp, query_heads=64 // tp, kv_heads=4 // tp,
               dtype=dtype, q_dtype=str(case["q"].dtype), pdl=pdl, seq_len=8192,
               topk=16, page=128, dql=1, seed=0, splits=calls[0][3]["NUM_TOPK_CHUNKS"],
               compiled=compiled, correctness=correctness,
               partial_graph=graph_time(partial), merge_graph=graph_time(merge),
               chain_graph=graph_time(chain), public_api=api_time(case))
    dump(directory / "measurement.json", row)
    print(json.dumps({k: row[k] for k in ("batch", "tp", "dtype", "pdl", "splits", "correctness")}
                     | {k: row[k]["median_us"] for k in ("partial_graph", "merge_graph", "chain_graph")}), flush=True)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("bench", "profile"), default="bench")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--dtype", choices=("bf16", "fp8"), default="bf16")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--pdl", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("/tmp/c2-artifacts"))
    args = parser.parse_args()
    prop = torch.cuda.get_device_properties(0)
    environment = dict(torch=torch.__version__, triton=triton.__version__, cuda=torch.version.cuda,
                       python=platform.python_version(), gpu=str(prop), capability=torch.cuda.get_device_capability(),
                       gpu_properties={key: getattr(prop, key, None) for key in
                                       ("name", "multi_processor_count", "total_memory", "shared_memory_per_block",
                                        "shared_memory_per_multiprocessor", "regs_per_multiprocessor", "L2_cache_size")},
                       upstream_sha256=hashlib.sha256((ROOT / "vllm_msa_ref/sparse_attn.py").read_bytes()).hexdigest())
    dump(args.output / "environment.json", environment)
    shapes = [(b, tp, dtype) for tp in (1, 4) for b in (1, 4, 8, 16) for dtype in ("bf16", "fp8")] if args.all else [(args.batch, args.tp, args.dtype)]
    rows = []
    for batch, tp, dtype in shapes:
        name = f"tp{tp}-b{batch}-{dtype}-pdl{int(args.pdl)}"
        row = run_one(batch, tp, dtype, args.output / name, args.pdl, args.mode == "profile")
        rows.append(row)
        dump(args.output / "measurements.json", rows)
    print("C2_BASELINE_DONE", flush=True)


if __name__ == "__main__":
    main()
