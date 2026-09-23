"""Validate and compare rectangular GEMMs in one B300 process.

The CPU build directory contains ``<case_name>__<variant>.so`` modules.
Each case is validated and measured before the next case is allocated.
"""

import argparse
import contextlib
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
import traceback


LABELS = {
    "release": "Four input stages, early release after final TMEM chunk read",
    "k128": "K tile 128 and two input stages, same total input shared memory",
    "original": "Original v9: four chunked epilogue stores",
    "fenced": "Original v9 with explicit TMEM cross-thread fences",
    "wide": "Wider epilogue stores",
    "full_late": "Full epilogue store, late accumulator release",
    "full_early": "Full epilogue store, early accumulator release",
    "cublas": "cuBLAS via torch.mm",
}


def write_json(path, obj):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def telemetry():
    result = {"time_utc": datetime.now(timezone.utc).isoformat()}
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,uuid,driver_version,pstate,"
             "temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,"
             "utilization.gpu,memory.used", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        )
        result.update(returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
    except Exception:
        result["error"] = traceback.format_exc()
    return result


def finite_scalar(tensor):
    value = tensor.item()
    return value if math.isfinite(value) else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    req = json.loads(args.request.read_text())
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    result = {"status": "starting", "request": req, "cases": {}}

    def checkpoint():
        write_json(out / "results.json", result)

    checkpoint()
    stream_context = contextlib.ExitStack()
    active_case = None
    try:
        import torch
        import tvm
        import tvm_ffi
        import tvm.support.nvcc  # Register load-time CUDA compilation callback.

        seeds = req.get("seeds", [0, 1, 2])
        if not seeds:
            raise ValueError("At least one correctness seed is required")
        if req["mode"] not in ("bench", "verify"):
            raise ValueError("mode must be bench or verify")
        if req["rounds"] < 1 or req["repeat"] < 1:
            raise ValueError("rounds and repeat must be positive")
        if len({case["name"] for case in req["cases"]}) != len(req["cases"]):
            raise ValueError("Case names must be unique")
        torch.set_num_threads(4)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cuda.preferred_blas_library("cublas")
        prop = torch.cuda.get_device_properties(0)
        capability = torch.cuda.get_device_capability()
        if "B300" not in prop.name:
            raise RuntimeError(f"Requested B300, got {prop.name}")
        if capability != (10, 3):
            raise RuntimeError(f"Unexpected compute capability {capability} for sm_103a build")
        torch.cuda.set_stream(torch.cuda.default_stream())
        stream_context.enter_context(tvm_ffi.use_torch_stream())
        packages = {}
        for name in ("torch", "apache-tvm", "apache-tvm-ffi", "cuda-bindings", "triton",
                     "numpy", "nvidia-cublas", "nvidia-cuda-runtime"):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        result["environment"] = {
            "gpu": prop.name, "sm_count": prop.multi_processor_count,
            "compiled_sm_count": req["sm_count"],
            "sm_count_matches_build": prop.multi_processor_count == req["sm_count"],
            "compute_capability": list(capability), "arch": req["arch"],
            "memory_bytes": prop.total_memory, "properties": str(prop),
            "python": sys.version, "torch_cuda": torch.version.cuda, "packages": packages,
            "cuda_stream": torch.cuda.current_stream().cuda_stream,
            "clock_policy": "Default dynamic clocks; no clock or power changes",
            "initial_telemetry": telemetry(),
            "cublas": {
                "interface": "torch.mm(A, B.T, out=D)",
                "preferred_blas_library": str(torch.backends.cuda.preferred_blas_library()),
                "allow_tf32": False,
                "allow_fp16_reduced_precision_reduction": False,
                "algorithm": "PyTorch/cuBLAS default selection, not explicitly tuned",
                "workspace": "PyTorch default, not explicitly controlled",
            },
        }
        result["protocol"] = {
            "operation": "D[M,N] = A[M,K] @ B[N,K].T",
            "shape_order": ["M", "N", "K"],
            "dtype": "FP16 input/output, FP32 accumulation",
            "reference": "torch FP32 GEMM with TF32 disabled, rounded to FP16",
            "rtol": 0.02, "atol": 0.01, "seeds": seeds,
            "correctness": "Full matrix checked for each seed; output filled with NaN before each call",
            "timer": "CUDA events on default stream, one interval per complete GEMM",
            "cuda_graphs": False,
            "excludes": ["compilation", "allocation", "input generation", "reference",
                         "correctness checking", "256 MiB cache flush"],
            "cache": "256 MiB buffer zeroed before every measured invocation, outside events",
            "warmup": "At least 5 calls and about 200 ms, capped at 500 calls",
            "rounds": req["rounds"],
            "repeat": "per-round count min(requested, max(5, ceil(200ms/calibration_ms)))",
            "statistic": "median of round means; individual event times and round means saved",
            "host_submission_gaps": "CUDA event intervals may include host submission gaps",
            "case_order": [case["name"] for case in req["cases"]],
            "round_order": "Independent deterministic shuffle of all variants and cuBLAS each round",
            "tuning": "Explicit requested candidates; no automatic candidate selection",
            "include_cublas": req.get("include_cublas", True),
            "historical_baseline_run": req.get("historical_baseline_run"),
            "baseline_remeasurement": "No: user requested reuse of already completed experiments",
        }
        result["status"] = "running"
        checkpoint()
        flush = (torch.empty(256 * 1024 * 1024, device="cuda", dtype=torch.uint8)
                 if req["mode"] == "bench" else None)

        def run_case(case, data):
            m, n, k = case["shape"]
            if any(not isinstance(x, int) or x <= 0 for x in (m, n, k)):
                raise ValueError(f"Invalid shape for {case['name']}: {case['shape']}")
            requested = case["variants"]
            if not requested or len(set(requested)) != len(requested) or "cublas" in requested:
                raise ValueError("variants must be unique kernel names, excluding implicit cublas")
            variants = requested + (["cublas"] if req.get("include_cublas", True) else [])
            a = torch.empty((m, k), device="cuda", dtype=torch.float16)
            b = torch.empty((n, k), device="cuda", dtype=torch.float16)
            outputs = {v: torch.empty((m, n), device="cuda", dtype=torch.float16) for v in variants}
            data["input_strides"] = {"A": list(a.stride()), "B": list(b.stride())}
            data["build_sha256"] = {}
            modules, functions = {}, {}
            for variant in requested:
                library = args.build / f"{case['name']}__{variant}.so"
                data["build_sha256"][library.name] = hashlib.sha256(library.read_bytes()).hexdigest()
                # CPU-exported fallback modules contain source; retain sm_103a
                # during the JIT performed when the runtime module is loaded.
                with tvm.target.Target({"kind": "cuda", "arch": req["arch"]}):
                    modules[variant] = tvm.runtime.load_module(str(library))
                functions[variant] = modules[variant].get_function("main", query_imports=True)
                print("LOADED", case["name"], variant, flush=True)
            functions["cublas"] = lambda x, y, z: torch.mm(x, y.T, out=z)
            calls = {v: (lambda v=v: functions[v](a, b, outputs[v])) for v in variants}
            checkpoint()

            for seed in seeds:
                torch.manual_seed(seed)
                a.normal_()
                b.normal_()
                expected = (a.float() @ b.float().T).half()
                reference = expected.float()
                torch.cuda.synchronize()
                for variant in variants:
                    outputs[variant].fill_(float("nan"))
                    calls[variant]()
                    torch.cuda.synchronize()
                    actual = outputs[variant].float()
                    error = (actual - reference).abs()
                    bound = 0.01 + 0.02 * reference.abs()
                    valid = torch.isfinite(actual) & (error <= bound)
                    record = {
                        "seed": seed, "passed": bool(valid.all().item()),
                        "max_abs_error": finite_scalar(error.max()),
                        "rms_error": finite_scalar(error.square().mean().sqrt()),
                        "relative_l2_error": finite_scalar(error.norm() / reference.norm()),
                        "mismatch_count": int((~valid).sum().item()),
                        "nonfinite_count": int((~torch.isfinite(actual)).sum().item()),
                    }
                    data["validation"].setdefault(variant, []).append(record)
                    print("VALIDATION", case["name"], variant, json.dumps(record), flush=True)
                    checkpoint()
                    if not record["passed"]:
                        raise RuntimeError(f"{case['name']}/{variant} failed correctness at seed {seed}")
                    del actual, error, bound, valid
                del expected, reference
            data["status"] = "validated"
            checkpoint()
            if req["mode"] == "verify":
                return

            def timed_batch(variant, count):
                starts = [torch.cuda.Event(enable_timing=True) for _ in range(count)]
                ends = [torch.cuda.Event(enable_timing=True) for _ in range(count)]
                for start, end in zip(starts, ends):
                    start.record()
                    end.record()
                torch.cuda.synchronize()
                for start, end in zip(starts, ends):
                    flush.zero_()
                    start.record()
                    calls[variant]()
                    end.record()
                ends[-1].synchronize()
                return [start.elapsed_time(end) for start, end in zip(starts, ends)]

            for variant in variants:
                estimate = statistics.median(timed_batch(variant, 3))
                calibrated_count = max(5, math.ceil(200 / max(estimate, 0.001)))
                warmup_count = min(500, calibrated_count)
                for _ in range(warmup_count):
                    calls[variant]()
                torch.cuda.synchronize()
                count = min(req["repeat"], calibrated_count)
                data["measurements"][variant] = {
                    "label": LABELS.get(variant, variant), "calibration_ms": estimate,
                    "warmup_calls": warmup_count, "calls_per_round": count,
                    "samples_ms": [], "round_means_ms": [],
                }
                print("WARMUP", case["name"], variant, estimate, warmup_count, count, flush=True)
                checkpoint()

            for round_index in range(req["rounds"]):
                order = variants.copy()
                random.Random(20260918 + round_index).shuffle(order)
                data["round_order"].append(order)
                time.sleep(0.25)
                data["telemetry"].append(telemetry())
                for variant in order:
                    calls[variant]()
                    torch.cuda.synchronize()
                    item = data["measurements"][variant]
                    samples = timed_batch(variant, item["calls_per_round"])
                    mean = statistics.mean(samples)
                    item["samples_ms"].append(samples)
                    item["round_means_ms"].append(mean)
                    print("ROUND", case["name"], round_index, variant, mean, flush=True)
                    checkpoint()
            for variant in variants:
                item = data["measurements"][variant]
                means = item["round_means_ms"]
                item["median_ms"] = statistics.median(means)
                item["mean_ms"] = statistics.mean(means)
                item["min_round_ms"] = min(means)
                item["max_round_ms"] = max(means)
                item["round_cv_percent"] = (100 * statistics.stdev(means) / statistics.mean(means)
                                            if len(means) > 1 else 0.0)
                item["tflops"] = 2 * m * n * k / item["median_ms"] / 1e9
            for item in data["measurements"].values():
                baseline = data["measurements"].get("original")
                item["relative_original"] = (baseline["median_ms"] / item["median_ms"]
                                             if baseline else None)
                cublas_measurement = data["measurements"].get("cublas")
                item["cublas_throughput_ratio"] = (cublas_measurement["median_ms"] / item["median_ms"]
                                                     if cublas_measurement else None)

            # Check the output left by repeated timed calls as well as the
            # isolated calls above, without adding validation to any interval.
            reference = (a.float() @ b.float().T).half().float()
            data["post_timing_validation"] = {}
            for variant in variants:
                actual = outputs[variant].float()
                error = (actual - reference).abs()
                valid = torch.isfinite(actual) & (error <= 0.01 + 0.02 * reference.abs())
                record = {"seed": seeds[-1], "passed": bool(valid.all().item()),
                          "mismatch_count": int((~valid).sum().item()),
                          "max_abs_error": finite_scalar(error.max())}
                data["post_timing_validation"][variant] = record
                checkpoint()
                if not record["passed"]:
                    raise RuntimeError(f"{case['name']}/{variant} failed after repeated timed calls")

        for case in req["cases"]:
            active_case = case["name"]
            data = {
                "status": "starting", "shape": case["shape"],
                "validation": {}, "measurements": {}, "round_order": [],
                "telemetry": [telemetry()],
            }
            result["cases"][active_case] = data
            checkpoint()
            print("CASE_START", active_case, json.dumps(case), flush=True)
            run_case(case, data)
            data["telemetry"].append(telemetry())
            data["status"] = "complete"
            checkpoint()
            print("CASE_COMPLETE", active_case,
                  json.dumps({v: x["median_ms"] for v, x in data["measurements"].items()}), flush=True)
            active_case = None
            gc.collect()
            torch.cuda.empty_cache()
        result["status"] = "complete"
        result["environment"]["final_telemetry"] = telemetry()
        checkpoint()
        print("COMPLETE", json.dumps(list(result["cases"])), flush=True)
    except BaseException:
        error = traceback.format_exc()
        result["status"] = "failed"
        result["error"] = error
        if active_case is not None and active_case in result["cases"]:
            result["cases"][active_case]["status"] = "failed"
            result["cases"][active_case]["error"] = error
        checkpoint()
        print("FAILED", active_case, error, flush=True)
        raise
    finally:
        stream_context.close()


if __name__ == "__main__":
    main()
