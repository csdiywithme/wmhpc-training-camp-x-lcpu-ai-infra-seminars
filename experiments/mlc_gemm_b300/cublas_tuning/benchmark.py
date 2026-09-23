"""T1--T3 only: strict FP32-reduction cuBLAS tuning and new large shapes.

No legacy benchmark, custom kernel, profiler, workspace sweep, or graph test is
launched here. Each completed candidate/round is checkpointed; final selection
is followed by independent validation and measurement.
"""
import ctypes as ct
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
import traceback


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def telemetry():
    record = {"time_utc": datetime.now(timezone.utc).isoformat()}
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,uuid,driver_version,pstate,temperature.gpu,"
             "power.draw,power.limit,clocks.sm,clocks.mem,utilization.gpu,memory.used",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=15)
        record.update(returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
    except Exception as exc:
        record["error"] = str(exc)
    return record


class Native:
    def __init__(self, path):
        self.lib = ct.CDLL(str(path))
        signatures = {
            "gemm_create": ([ct.c_int64] * 3 + [ct.c_uint64] * 5, ct.c_void_p),
            "gemm_candidate_count": ([ct.c_void_p], ct.c_int),
            "gemm_candidate_json": ([ct.c_void_p, ct.c_int], ct.c_char_p),
            "gemm_run_lt": ([ct.c_void_p, ct.c_int], ct.c_int),
            "gemm_run_autotune": ([ct.c_void_p], ct.c_int),
            "gemm_error": ([ct.c_void_p], ct.c_char_p),
            "gemm_info_json": ([ct.c_void_p], ct.c_char_p),
            "gemm_destroy": ([ct.c_void_p], None),
        }
        for name, (args, restype) in signatures.items():
            func = getattr(self.lib, name)
            func.argtypes, func.restype = args, restype

    def error(self, handle):
        text = self.lib.gemm_error(handle)
        return text.decode() if text else "No native error details"

    def execute_lt(self, handle, index):
        if self.lib.gemm_run_lt(handle, index):
            raise RuntimeError(self.error(handle))

    def execute_autotune(self, handle):
        if self.lib.gemm_run_autotune(handle):
            raise RuntimeError(self.error(handle))


def run(request, output_path, library_path, checkpoint_callback=None):
    import torch

    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    library_path = Path(library_path)
    previous = out / "results.json"
    digest = hashlib.sha256(library_path.read_bytes()).hexdigest()
    if previous.exists():
        result = json.loads(previous.read_text())
        if result["request"] != request or result["library_sha256"] != digest:
            raise RuntimeError("Refusing to reuse results with different request/native binary")
    else:
        result = {"status": "starting", "request": request, "library_sha256": digest,
                  "cases": {}, "sessions": []}

    def checkpoint(force=False):
        write_json(previous, result)
        if checkpoint_callback:
            checkpoint_callback(force=force)

    if set(request["scope"]) != {"T1", "T2", "T3"}:
        raise ValueError("This runner supports exactly T1, T2, T3")
    if request["requested_algorithms"] != 32 or request["workspace_bytes"] != 64 * 1024**2:
        raise ValueError("T1--T3 uses exactly 32 requested candidates and 64 MiB workspace")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.preferred_blas_library("cublas")
    torch.cuda.set_stream(torch.cuda.default_stream())
    props = torch.cuda.get_device_properties(0)
    if "B300" not in props.name or torch.cuda.get_device_capability() != (10, 3):
        raise RuntimeError(f"Unexpected GPU: {props}")
    packages = {}
    for name in ("torch", "numpy", "nvidia-cublas", "nvidia-cuda-runtime"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    if packages["nvidia-cublas"] != request["expected_cublas_package"]:
        raise RuntimeError(f"Unexpected cuBLAS package: {packages['nvidia-cublas']}")
    native = Native(library_path)
    # Keep library resolutions as well as package versions: those can differ.
    loaded_libraries = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                               if "libcublas" in line or "libcudart" in line})
    session = {"time_utc": datetime.now(timezone.utc).isoformat(), "gpu": props.name,
               "sm_count": props.multi_processor_count, "properties": str(props),
               "memory_bytes": props.total_memory, "compute_capability": [10, 3],
               "python": sys.version, "torch_cuda": torch.version.cuda,
               "packages": packages, "loaded_libraries": loaded_libraries,
               "initial_telemetry": telemetry()}
    result["sessions"].append(session)
    session_index = len(result["sessions"]) - 1
    result["protocol"] = {
        "operation": "D[M,N] = A[M,K] @ B[N,K].T; alpha=1, beta=0",
        "dtype": "FP16 input/output; FP32 accumulation; no FP16 intermediate reduction",
        "lt_precision": "compute32F; reduction mask COMPUTE_TYPE; inspect candidate reduction/accumulator flags",
        "gemmex_precision": "compute32F; CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION",
        "torch_precision": "allow_tf32=False; allow_fp16_reduced_precision_reduction=False",
        "reference": "same FP16 inputs converted to FP32; torch.mm with TF32 off; rounded to FP16",
        "correctness": "all candidates seed0; selected methods seeds0/1/2 and after formal timing; full matrix, NaN prefill",
        "timer": "CUDA events, one interval per GEMM, default stream; host submission gaps may be included",
        "statistic": "median of round means; raw event samples saved",
        "tuning": "3 shuffled rounds x 10 calls per candidate, disjoint from final5round measurement",
        "warmup": "tuning10 calls; formal at least5 and about200ms, capped500 calls",
        "flush256m": "zero256MiB buffer before every timed call outside CUDA event interval",
        "steady": "same buffers reused continuously, no explicit cache flush; not guaranteed all data cache-resident",
        "clock_policy": "default dynamic clocks/power; no changes",
        "cuda_graphs": False, "ncu": False,
        "excludes": ["compilation", "allocation", "input generation", "reference", "validation",
                     "heuristics", "AUTOTUNE first call", "warmup", "cache flush"],
        "historical_comparisons": "Cross-session saved baselines only; no old default or custom kernel reruns",
    }
    result["status"] = "running"
    checkpoint(True)
    flush = torch.empty(256 * 1024**2, device="cuda", dtype=torch.uint8)
    active_case = None

    def scalar(value):
        value = float(value.item())
        return value if math.isfinite(value) else None

    def validate(output, reference, seed):
        # Chunk rows to bound temporary memory on 16384^2 outputs; every element
        # participates in the check, including near-zero cancellation results.
        max_abs, square_sum, reference_square_sum = 0., 0., 0.
        mismatch, nonfinite = 0, 0
        count = output.numel()
        for start in range(0, output.shape[0], 1024):
            actual = output[start:start + 1024].float()
            target = reference[start:start + 1024].float()
            error = (actual - target).abs()
            finite = torch.isfinite(actual)
            valid = finite & (error <= request["atol"] + request["rtol"] * target.abs())
            nonfinite += int((~finite).sum().item())
            mismatch += int((~valid).sum().item())
            maximum = scalar(error.max())
            if maximum is not None:
                max_abs = max(max_abs, maximum)
            value = scalar(error.square().sum(dtype=torch.float64))
            square_sum += value if value is not None else 0
            reference_square_sum += float(target.square().sum(dtype=torch.float64).item())
        return {"seed": seed, "passed": mismatch == 0, "mismatch_count": mismatch,
                "nonfinite_count": nonfinite, "max_abs_error": max_abs if nonfinite == 0 else None,
                "rms_error": math.sqrt(square_sum/count) if nonfinite == 0 else None,
                "relative_l2_error": math.sqrt(square_sum / reference_square_sum)
                   if nonfinite == 0 and reference_square_sum else None}

    def timed(call, count, cache):
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(count)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(count)]
        # Initialize events outside measurement, matching the prior benchmark.
        for start, end in zip(starts, ends):
            start.record()
            end.record()
        torch.cuda.synchronize()
        for start, end in zip(starts, ends):
            if cache == "flush256m":
                flush.zero_()
            start.record()
            call()
            end.record()
        ends[-1].synchronize()
        return [start.elapsed_time(end) for start, end in zip(starts, ends)]

    def warmup(call, count):
        for _ in range(count):
            call()
        torch.cuda.synchronize()

    try:
        for case in request["cases"]:
            name = active_case = case["name"]
            data = result["cases"].setdefault(name, {"spec": case, "status": "starting",
                                                     "candidates": {}, "measurements": {},
                                                     "validation": {}, "telemetry": []})
            if data["spec"] != case:
                raise RuntimeError("Case metadata changed during resume")
            if data["status"] == "complete":
                print("SKIP_COMPLETE", name, flush=True)
                continue
            data["session_index"] = session_index
            data.setdefault("sessions_used", []).append(session_index)
            data["telemetry"].append(telemetry())
            m, n, k = case["shape"]
            a = torch.empty((m, k), device="cuda", dtype=torch.float16)
            b = torch.empty((n, k), device="cuda", dtype=torch.float16)
            d = torch.empty((m, n), device="cuda", dtype=torch.float16)
            handle = native.lib.gemm_create(m, n, k, request["workspace_bytes"],
                                            torch.cuda.current_stream().cuda_stream,
                                            a.data_ptr(), b.data_ptr(), d.data_ptr())
            if not handle:
                raise RuntimeError(native.error(None))
            try:
                info = json.loads(native.lib.gemm_info_json(handle).decode())
                data["native"] = info
                count = native.lib.gemm_candidate_count(handle)
                if count < 1:
                    raise RuntimeError(f"No strict-precision cuBLASLt candidates for {name}")
                torch.manual_seed(0)
                a.normal_()
                b.normal_()
                reference = torch.mm(a.float(), b.float().T).half()
                torch.cuda.synchronize()
                calls = {}
                for index in range(count):
                    config = json.loads(native.lib.gemm_candidate_json(handle, index).decode())
                    item = data["candidates"].setdefault(str(index), {"config": config,
                             "status": "starting", "samples_ms": [], "round_means_ms": []})
                    if item["config"] != config:
                        raise RuntimeError("Candidate list/config changed during resume; not repeating saved work")
                    call = lambda index=index: native.execute_lt(handle, index)
                    calls[index] = call
                    if item["status"] in ("rejected", "complete") or "validation" in item:
                        continue
                    try:
                        d.fill_(float("nan"))
                        call()
                    except RuntimeError as exc:
                        # A synchronous unsupported algorithm can be skipped;
                        # an asynchronous/device fault below is fatal.
                        torch.cuda.synchronize()
                        item.update(status="rejected", reason=str(exc))
                        checkpoint()
                        continue
                    torch.cuda.synchronize()
                    record = validate(d, reference, 0)
                    item["validation"] = record
                    if not record["passed"]:
                        item.update(status="rejected", reason="full-matrix correctness failed")
                    else:
                        item["status"] = "validated"
                    print("CANDIDATE_VALIDATED", name, index, record["passed"], flush=True)
                    checkpoint()

                valid_indices = [i for i in range(count) if data["candidates"][str(i)]["status"] != "rejected"]
                if not valid_indices:
                    raise RuntimeError(f"All cuBLASLt candidates failed validation for {name}")
                # Train only once. Resume skips each already-completed round.
                for round_index in range(request["tune_rounds"]):
                    order = valid_indices.copy()
                    random.Random(617 + round_index).shuffle(order)
                    for index in order:
                        item = data["candidates"][str(index)]
                        if len(item["samples_ms"]) > round_index:
                            continue
                        warmup(calls[index], request["tune_warmup"])
                        samples = timed(calls[index], request["tune_repeat"], case["cache"])
                        item["samples_ms"].append(samples)
                        item["round_means_ms"].append(statistics.mean(samples))
                        item.setdefault("round_session_indices", []).append(session_index)
                        if len(item["samples_ms"]) == request["tune_rounds"]:
                            item["tuning_ms"] = statistics.median(item["round_means_ms"])
                            item["status"] = "complete"
                        print("TUNE_ROUND", name, index, round_index, statistics.mean(samples), flush=True)
                        checkpoint()
                selected = min(valid_indices, key=lambda i: data["candidates"][str(i)]["tuning_ms"])
                data["selected_candidate"] = selected
                data["selection"] = {"criterion": "lowest median of tuning round means among validated candidates",
                                     "requested": 32, "accepted": count, "correct_candidates": len(valid_indices),
                                     "config": data["candidates"][str(selected)]["config"],
                                     "training_ms": data["candidates"][str(selected)]["tuning_ms"]}
                methods = {"lt_tuned": calls[selected]}
                if "gemmex_autotune" in case["methods"]:
                    def autotune():
                        status = native.lib.gemm_run_autotune(handle)
                        if status:
                            raise RuntimeError(native.error(handle))
                    # Explicitly isolate the first internal search from timing.
                    torch.cuda.synchronize()
                    first_start = time.perf_counter()
                    autotune()
                    torch.cuda.synchronize()
                    data.setdefault("autotune_initializations", []).append({
                        "session_index": session_index, "wall_seconds": time.perf_counter() - first_start,
                        "excluded_from_formal_measurement": True})
                    methods["gemmex_autotune"] = autotune
                if "torch_default" in case["methods"]:
                    methods["torch_default"] = lambda: torch.mm(a, b.T, out=d)
                if set(methods) != set(case["methods"]):
                    raise RuntimeError("Unexpected benchmark method")
                print("SELECTED", name, selected, json.dumps(data["selection"]), flush=True)
                checkpoint(True)

                for seed in request["seeds"]:
                    if seed != 0:
                        torch.manual_seed(seed)
                        a.normal_()
                        b.normal_()
                        reference = torch.mm(a.float(), b.float().T).half()
                        torch.cuda.synchronize()
                    for method, call in methods.items():
                        records = data["validation"].setdefault(method, [])
                        saved = [record for record in records if record["seed"] == seed]
                        if any(not record["passed"] for record in saved):
                            raise RuntimeError(f"Saved correctness failure for {name}/{method}/seed{seed}; refusing resume")
                        if saved:
                            continue
                        d.fill_(float("nan"))
                        call()
                        torch.cuda.synchronize()
                        record = validate(d, reference, seed)
                        record["session_index"] = session_index
                        records.append(record)
                        print("VALIDATION", name, method, json.dumps(record), flush=True)
                        checkpoint()
                        if not record["passed"]:
                            raise RuntimeError(f"Selected {name}/{method} failed correctness; refusing performance claim")

                for method, call in methods.items():
                    item = data["measurements"].setdefault(method, {"samples_ms": [], "round_means_ms": []})
                    estimate = statistics.median(timed(call, 3, case["cache"]))
                    calibrated = max(5, math.ceil(200 / max(estimate, .001)))
                    warm = min(500, calibrated)
                    warmup(call, warm)
                    item.setdefault("calibration_ms", estimate)
                    item.setdefault("warmup_calls", warm)
                    item.setdefault("calls_per_round", min(request["repeat"], calibrated))
                    checkpoint()
                for round_index in range(request["rounds"]):
                    order = list(methods)
                    random.Random(20260920 + round_index).shuffle(order)
                    data["telemetry"].append(telemetry())
                    for method in order:
                        item = data["measurements"][method]
                        if len(item["samples_ms"]) > round_index:
                            continue
                        methods[method]()
                        torch.cuda.synchronize()
                        samples = timed(methods[method], item["calls_per_round"], case["cache"])
                        item["samples_ms"].append(samples)
                        item["round_means_ms"].append(statistics.mean(samples))
                        item.setdefault("round_session_indices", []).append(session_index)
                        print("MEASURE_ROUND", name, method, round_index, item["round_means_ms"][-1], flush=True)
                        checkpoint()
                for method, call in methods.items():
                    item = data["measurements"][method]
                    means = item["round_means_ms"]
                    item["median_ms"] = statistics.median(means)
                    item["tflops"] = 2 * m * n * k / (item["median_ms"] * 1e9)
                    item["peak_percent"] = 100 * item["tflops"] / request["dense_peak_tflops"]
                    item["round_cv_percent"] = 100 * statistics.stdev(means) / statistics.mean(means)
                    d.fill_(float("nan"))
                    call()
                    torch.cuda.synchronize()
                    item["post_validation"] = validate(d, reference, request["seeds"][-1])
                    if not item["post_validation"]["passed"]:
                        raise RuntimeError(f"{name}/{method} failed post-timing validation")
                    print("ROW", json.dumps({"case": name, "method": method, "cache": case["cache"],
                          **{key: item[key] for key in ("median_ms", "tflops", "peak_percent", "round_cv_percent")}}), flush=True)
                data["status"] = "complete"
                data["telemetry"].append(telemetry())
                checkpoint(True)
                print("CASE_COMPLETE", name, flush=True)
                del reference, calls, methods, a, b, d
            finally:
                native.lib.gemm_destroy(handle)
            gc.collect()
            torch.cuda.empty_cache()
        result["status"] = "complete"
        result["completed_utc"] = datetime.now(timezone.utc).isoformat()
        result["final_telemetry"] = telemetry()
        checkpoint(True)
        return result
    except BaseException:
        result["status"] = "failed"
        result["error"] = traceback.format_exc()
        result["active_case"] = active_case
        if active_case:
            result["cases"][active_case]["status"] = "failed"
        checkpoint(True)
        raise


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.request.read_text()), args.output, args.library)
