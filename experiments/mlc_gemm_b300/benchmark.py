"""Validate all outputs, then interleave CUDA-event measurements on one GPU."""
import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import time
import traceback

LABELS = {1: "Sync load + MMA (full-matrix adaptation)",
          2: "K-loop TMEM accumulation (serial output tiles)",
          3: "Spatial tiling (multi-CTA)", 4: "TMA async load",
          5: "Software pipeline", 6: "Persistent kernel",
          7: "Warp specialization", 8: "Two-CTA cluster",
          9: "Multi-consumer", 0: "cuBLAS via torch.mm"}

def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")

def telemetry():
    p = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version,pstate,"
                        "temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,"
                        "utilization.gpu,memory.used", "--format=csv,noheader"],
                       capture_output=True, text=True, timeout=15)
    return {"time_utc": datetime.now(timezone.utc).isoformat(),
            "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    req = json.loads(args.request.read_text())
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    result = {"request": req, "status": "starting", "validation": {},
              "measurements": {}, "telemetry": [], "round_order": []}

    def checkpoint():
        write_json(out / "results.json", result)

    stream_context = contextlib.ExitStack()
    try:
        import torch
        import tvm
        import tvm_ffi
        import tvm.support.nvcc  # register the CUDA compilation callback
        torch.set_num_threads(4)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cuda.preferred_blas_library("cublas")
        prop = torch.cuda.get_device_properties(0)
        if "B300" not in prop.name:
            raise RuntimeError(f"Requested B300, got {prop.name}")
        if prop.multi_processor_count != req["sm_count"]:
            raise RuntimeError(f"Recompile with actual SM count {prop.multi_processor_count}")
        if torch.cuda.get_device_capability() != (10, 3):
            raise RuntimeError("Unexpected compute capability for sm_103a build")
        # All library calls and CUDA events use stream zero. No work uses an
        # auxiliary stream; explicit synchronization precedes each timing batch.
        torch.cuda.set_stream(torch.cuda.default_stream())
        stream_context.enter_context(tvm_ffi.use_torch_stream())
        versions = req["versions"] + [0]
        packages = {}
        for name in ("torch", "apache-tvm", "apache-tvm-ffi", "cuda-bindings", "triton",
                     "numpy", "nvidia-cublas", "nvidia-cuda-runtime"):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        result["environment"] = {
            "gpu": prop.name, "sm_count": prop.multi_processor_count,
            "compute_capability": list(torch.cuda.get_device_capability()),
            "memory_bytes": prop.total_memory, "properties": str(prop),
            "torch_cuda": torch.version.cuda, "packages": packages,
            "cuda_stream": torch.cuda.current_stream().cuda_stream,
            "clock_policy": "Default dynamic clocks; no clock or power changes",
            "cublas": {"interface": "torch.mm(A, B.T, out=D)",
                       "preferred_blas_library": str(torch.backends.cuda.preferred_blas_library()),
                       "allow_fp16_reduced_precision_reduction": False,
                       "algorithm": "PyTorch/cuBLAS default selection, not explicitly tuned",
                       "workspace": "PyTorch default, not explicitly controlled"}}
        result["telemetry"].append(telemetry())
        result["protocol"] = {
            "operation": "D = A @ B.T", "shape": [req["size"]] * 3,
            "dtype": "FP16 input/output, FP32 accumulation",
            "reference": "torch FP32 GEMM with TF32 disabled, rounded to FP16",
            "rtol": 0.02, "atol": 0.01, "seeds": [0, 1, 2],
            "timer": "CUDA events on default stream, one interval per complete GEMM",
            "excludes": ["compilation", "allocation", "input generation", "reference",
                         "correctness checking", "256 MiB cache flush"],
            "cache": "256 MiB buffer zeroed before every measured invocation, outside events",
            "warmup": "At least 5 calls and about 200 ms, capped at 500 calls",
            "rounds": req["rounds"],
            "repeat": "per-round count min(requested, max(5, ceil(200ms/calibration_ms)))",
            "statistic": "median of round means; individual event times and round means saved",
            "host_submission_gaps": "CUDA event intervals may include host submission gaps",
            "tuning": "Tutorial parameters retained, no per-version autotuning",
        }
        n = req["size"]
        a = torch.empty((n, n), device="cuda", dtype=torch.float16)
        b = torch.empty_like(a)
        result["protocol"]["input_strides"] = {"A": list(a.stride()), "B": list(b.stride())}
        outputs = {v: torch.empty_like(a) for v in versions}
        modules = {}
        functions = {}
        for v in req["versions"]:
            library = args.build / f"v{v}.so"
            # CPU-exported CUDA fallback modules contain source, not a target.
            # Preserve the architecture-specific features during load-time JIT.
            with tvm.target.Target({"kind": "cuda", "arch": req["arch"]}):
                modules[v] = tvm.runtime.load_module(str(library))
            functions[v] = modules[v].get_function("main", query_imports=True)
        functions[0] = lambda x, y, z: torch.mm(x, y.T, out=z)
        calls = {v: (lambda v=v: functions[v](a, b, outputs[v])) for v in versions}
        result["build_sha256"] = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in args.build.glob("*.so")}
        checkpoint()

        # Full outputs, all variants, three shared seeded inputs. Fill with NaN
        # before every check so incomplete output coverage cannot pass silently.
        for seed in (0, 1, 2):
            torch.manual_seed(seed)
            a.normal_()
            b.normal_()
            expected = (a.float() @ b.float().T).half()
            torch.cuda.synchronize()
            for v in versions:
                outputs[v].fill_(float("nan"))
                calls[v]()
                torch.cuda.synchronize()
                actual = outputs[v].float()
                reference = expected.float()
                error = (actual - reference).abs()
                bound = 0.01 + 0.02 * reference.abs()
                valid = torch.isfinite(actual) & (error <= bound)
                record = {"seed": seed, "passed": bool(valid.all().item()),
                          "max_abs_error": error.max().item(),
                          "rms_error": error.square().mean().sqrt().item(),
                          "relative_l2_error": (error.norm() / reference.norm()).item(),
                          "mismatch_count": int((~valid).sum().item())}
                result["validation"].setdefault(str(v), []).append(record)
                print("VALIDATION", v, json.dumps(record), flush=True)
                checkpoint()
                if not record["passed"]:
                    raise RuntimeError(f"v{v} failed correctness at seed {seed}")
        result["status"] = "validated"
        checkpoint()
        if req["mode"] == "verify":
            result["status"] = "complete"
            checkpoint()
            return

        flush = torch.empty(256 * 1024 * 1024, device="cuda", dtype=torch.uint8)
        def timed_batch(v, count):
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(count)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(count)]
            # Initialize event handles before the measured round.
            for s, e in zip(starts, ends):
                s.record()
                e.record()
            torch.cuda.synchronize()
            for s, e in zip(starts, ends):
                flush.zero_()
                s.record()
                calls[v]()
                e.record()
            ends[-1].synchronize()
            return [s.elapsed_time(e) for s, e in zip(starts, ends)]

        for v in versions:
            estimate = statistics.median(timed_batch(v, 3))
            warmup_count = min(500, max(5, math.ceil(200 / max(estimate, 0.001))))
            for _ in range(warmup_count):
                calls[v]()
            torch.cuda.synchronize()
            count = min(req["repeat"], max(5, math.ceil(200 / max(estimate, 0.001))))
            result["measurements"][str(v)] = {
                "label": LABELS[v], "calibration_ms": estimate,
                "warmup_calls": warmup_count, "calls_per_round": count,
                "samples_ms": [], "round_means_ms": []}
            print("WARMUP", v, estimate, warmup_count, count, flush=True)
        for r in range(req["rounds"]):
            order = versions.copy()
            random.Random(20260918 + r).shuffle(order)
            result["round_order"].append(order)
            time.sleep(0.25)
            result["telemetry"].append(telemetry())
            for v in order:
                # Prime code paths after switching versions. Cache is flushed
                # separately before each recorded call, including the first.
                calls[v]()
                torch.cuda.synchronize()
                item = result["measurements"][str(v)]
                samples = timed_batch(v, item["calls_per_round"])
                mean = statistics.mean(samples)
                item["samples_ms"].append(samples)
                item["round_means_ms"].append(mean)
                print("ROUND", r, v, mean, flush=True)
                checkpoint()
        for v in versions:
            item = result["measurements"][str(v)]
            means = item["round_means_ms"]
            item["median_ms"] = statistics.median(means)
            item["mean_ms"] = statistics.mean(means)
            item["min_round_ms"] = min(means)
            item["max_round_ms"] = max(means)
            item["round_cv_percent"] = (100 * statistics.stdev(means) / statistics.mean(means)
                                        if len(means) > 1 else 0.0)
            item["tflops"] = 2 * n**3 / item["median_ms"] / 1e9
        for item in result["measurements"].values():
            if "1" in result["measurements"]:
                item["speedup_v1"] = result["measurements"]["1"]["median_ms"] / item["median_ms"]
            item["cublas_throughput_ratio"] = result["measurements"]["0"]["median_ms"] / item["median_ms"]
        result["telemetry"].append(telemetry())
        result["status"] = "complete"
        checkpoint()
        print("COMPLETE", json.dumps({v: x["median_ms"] for v, x in result["measurements"].items()}), flush=True)
    except BaseException:
        result["status"] = "failed"
        result["error"] = traceback.format_exc()
        checkpoint()
        raise
    finally:
        stream_context.close()

if __name__ == "__main__":
    main()
