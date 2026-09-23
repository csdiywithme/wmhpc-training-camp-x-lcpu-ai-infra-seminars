"""Balanced, fixed-shape comparison of the saved wide kernel and TMEM hoist.

This script never builds a kernel or starts a Modal job.  Run it inside the
experiment's B300 container, after build/hoisted.so has been created.  A separate
--profile-variant invocation selects exactly one candidate call with NVTX.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
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


SHAPE = (2048, 9472, 8192)
VARIANTS = ("baseline", "hoisted")
NVTX_RANGE = "tmem_hoist_candidate"
BASELINE_PATH = Path("/opt/tmem_hoist/input/aligned148_k8192__wide.so")
BASELINE_SHA256 = "292e320fc1373aa69b4df5e9ebaef849300c740774f8dfc11099ca464439b683"
EXPECTED_PACKAGES = {
    "torch": "2.10.0+cu130",
    "apache-tvm": "0.26.0",
    "apache-tvm-ffi": "0.1.14.post0",
    "nvidia-cublas": "13.1.0.3",
    "nvidia-cuda-runtime": "13.0.96",
}
SEEDS = (0, 1, 2)
BLOCKS = 12
SAMPLES_PER_SLOT = 300
WARMUP_CALLS = 50
ORDER_SEED = 20260920


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def start_result(path, value):
    """Even a failed prior run is immutable: there is no implicit resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def telemetry():
    fields = ["index", "uuid", "name", "driver_version", "pstate", "temperature.gpu",
              "power.draw", "power.limit", "clocks.sm", "clocks.mem",
              "utilization.gpu", "memory.used"]
    record = {"time_utc": utc_now()}
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=" + ",".join(fields),
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15)
        record.update(returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
        if proc.returncode == 0:
            record["gpus"] = [dict(zip(fields, [item.strip() for item in row]))
                              for row in csv.reader(io.StringIO(proc.stdout)) if row]
    except Exception as exc:
        record["error"] = str(exc)
    return record


def configure(torch):
    packages = {name: importlib.metadata.version(name) for name in EXPECTED_PACKAGES}
    if packages != EXPECTED_PACKAGES:
        raise RuntimeError(f"Packages differ from the original benchmark: {packages}")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.preferred_blas_library("cublas")
    torch.cuda.set_device(0)
    torch.cuda.set_stream(torch.cuda.default_stream(0))
    props = torch.cuda.get_device_properties(0)
    if ("B300" not in props.name or torch.cuda.get_device_capability(0) != (10, 3)
            or props.multi_processor_count != 148):
        raise RuntimeError(f"Expected B300 with 148 SMs, got {props}")
    initial_telemetry = telemetry()
    gpu_uuid = getattr(props, "uuid", None)
    gpu_uuid = str(gpu_uuid) if gpu_uuid is not None else None
    if not gpu_uuid:
        gpu_rows = initial_telemetry.get("gpus", [])
        if len(gpu_rows) == 1:
            gpu_uuid = gpu_rows[0]["uuid"]
        else:
            raise RuntimeError("Could not unambiguously record the tested GPU UUID")
    return {
        "packages": packages, "python": sys.version, "torch_cuda": torch.version.cuda,
        "gpu": {"name": props.name, "uuid": gpu_uuid,
                "sm_count": props.multi_processor_count,
                "compute_capability": [props.major, props.minor],
                "total_memory_bytes": props.total_memory, "properties": str(props)},
        "initial_telemetry": initial_telemetry,
    }


def library_info(path):
    path = path.resolve(strict=True)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size}


def load_kernel(tvm, path):
    with tvm.target.Target({"kind": "cuda", "arch": "sm_103a"}):
        module = tvm.runtime.load_module(str(path))
    # Keep the module alive for the lifetime of its exported function.
    return module, module.get_function("main", query_imports=True)


def validate(torch, output, reference, seed):
    maximum, square_sum, reference_square_sum = 0.0, 0.0, 0.0
    mismatch, nonfinite = 0, 0
    for start in range(0, output.shape[0], 1024):
        actual = output[start:start + 1024].float()
        target = reference[start:start + 1024].float()
        error = (actual - target).abs()
        finite = torch.isfinite(actual)
        valid = finite & (error <= 0.01 + 0.02 * target.abs())
        nonfinite += int((~finite).sum().item())
        mismatch += int((~valid).sum().item())
        chunk_maximum = float(error.max().item())
        if math.isfinite(chunk_maximum):
            maximum = max(maximum, chunk_maximum)
        chunk_square = float(error.square().sum(dtype=torch.float64).item())
        if math.isfinite(chunk_square):
            square_sum += chunk_square
        reference_square_sum += float(target.square().sum(dtype=torch.float64).item())
    return {
        "seed": seed, "passed": mismatch == 0, "checked_elements": output.numel(),
        "atol": 0.01, "rtol": 0.02, "row_chunk": 1024,
        "mismatch_count": mismatch, "nonfinite_count": nonfinite,
        "max_abs_error": maximum if nonfinite == 0 else None,
        "rms_error": math.sqrt(square_sum / output.numel()) if nonfinite == 0 else None,
        "relative_l2_error": math.sqrt(square_sum / reference_square_sum)
        if nonfinite == 0 and reference_square_sum else None,
    }


def timed_slot(torch, call):
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(SAMPLES_PER_SLOT)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(SAMPLES_PER_SLOT)]
    # Materialize lazily-created events before the equal 50-call warmup.
    for start, end in zip(starts, ends):
        start.record()
        end.record()
    for _ in range(WARMUP_CALLS):
        call()
    torch.cuda.synchronize()
    wall_start = time.perf_counter()
    for start, end in zip(starts, ends):
        start.record()
        call()
        end.record()
    ends[-1].synchronize()
    submission_and_wait_seconds = time.perf_counter() - wall_start
    samples = [float(start.elapsed_time(end)) for start, end in zip(starts, ends)]
    if not all(math.isfinite(value) and value > 0 for value in samples):
        raise RuntimeError("CUDA events returned invalid timing samples")
    return {"samples_ms": samples, "sample_count": len(samples),
            "mean_ms": statistics.fmean(samples), "median_ms": statistics.median(samples),
            "min_ms": min(samples), "max_ms": max(samples),
            "sample_cv_pct": 100 * statistics.stdev(samples) / statistics.fmean(samples),
            "submission_and_wait_seconds_diagnostic_only": submission_and_wait_seconds}


def quantile(values, probability):
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summarize(blocks):
    if not blocks:
        return {}
    flops = 2 * math.prod(SHAPE)
    by_variant = {}
    for variant in VARIANTS:
        values = [block["paired"][variant + "_mean_ms"] for block in blocks]
        median_ms = statistics.median(values)
        positions = {}
        for position in range(4):
            slots = [block["slots"][position]["mean_ms"] for block in blocks
                     if block["slots"][position]["variant"] == variant]
            if slots:
                positions[str(position + 1)] = {
                    "slot_count": len(slots), "slot_means_ms": slots,
                    "median_slot_mean_ms": statistics.median(slots),
                    "mean_slot_mean_ms": statistics.fmean(slots)}
        by_variant[variant] = {
            "block_means_ms": values, "median_block_mean_ms": median_ms,
            "median_block_mean_tflops": flops / median_ms / 1e9,
            "mean_block_mean_ms": statistics.fmean(values),
            "block_cv_pct": 100 * statistics.stdev(values) / statistics.fmean(values)
            if len(values) > 1 else None,
            "by_slot_position": positions,
        }
    ratios = [block["paired"]["baseline_over_hoisted"] for block in blocks]
    geometric_mean = math.exp(statistics.fmean([math.log(value) for value in ratios]))
    bootstrap = random.Random(ORDER_SEED + 1)
    # The experimental unit is an entire balanced block, not individual kernels.
    resampled = [math.exp(statistics.fmean(
        math.log(ratios[bootstrap.randrange(len(ratios))]) for _ in ratios))
        for _ in range(10000)] if len(ratios) > 1 else []
    interval = [quantile(resampled, 0.025), quantile(resampled, 0.975)] if resampled else None
    return {
        "completed_blocks": len(blocks), "variants": by_variant,
        "paired_block_ratios_baseline_over_hoisted": ratios,
        "paired_median_ratio": statistics.median(ratios),
        "paired_geometric_mean_ratio": geometric_mean,
        "paired_geometric_mean_speedup_pct": 100 * (geometric_mean - 1),
        "paired_geometric_mean_ratio_bootstrap_95pct_ci": interval,
        "bootstrap_unit": "balanced ABBA/BAAB block; 10000 resamples; descriptive within this single GPU session",
        "candidate_faster_blocks": sum(value > 1 for value in ratios),
        "patterns": {pattern: {
            "block_count": sum(block["pattern"] == pattern for block in blocks),
            "paired_ratios": [block["paired"]["baseline_over_hoisted"] for block in blocks
                              if block["pattern"] == pattern]}
            for pattern in ("ABBA", "BAAB")},
        "interpretation": (
            "Effect is below 3%; do not describe this run alone as a robust optimization."
            if abs(geometric_mean - 1) < 0.03 else
            "Assess block/order consistency and the confidence interval before claiming a speedup."),
        "caveat": "CUDA-event intervals can include GPU idle gaps caused by host submission; clocks are dynamic. No historical run is a paired control.",
    }


def benchmark(run_dir):
    path = run_dir / "benchmark" / "results.json"
    result = {
        "status": "starting", "started_at": utc_now(), "shape": list(SHAPE),
        "arch": "sm_103a", "operation": "D[M,N] = A[M,K] @ B[N,K].T",
        "validation": {"before": [], "after": [], "direct_comparison": []}, "blocks": [],
        "protocol": {
            "dtype": "FP16 inputs/output; FP32 accumulation",
            "reference": "same FP16 inputs converted to FP32; TF32 disabled; torch.mm rounded to FP16",
            "validation_seeds": list(SEEDS), "timing_seed": SEEDS[-1],
            "correctness": "all output elements, NaN prefill, both variants at seeds 0/1/2 before timing and seed 2 after timing",
            "block_count": BLOCKS, "slots_per_block": 4,
            "order": "six ABBA and six BAAB blocks, shuffled with fixed Python Random seed",
            "order_seed": ORDER_SEED, "A": "baseline", "B": "hoisted",
            "warmup_calls_per_slot": WARMUP_CALLS,
            "samples_per_slot": SAMPLES_PER_SLOT,
            "timer": "CUDA events on default stream, one pair per invocation, identical for both variants; host submission gaps may be included",
            "primary_statistic": "median of balanced block mean latencies and paired baseline/hoisted ratios",
            "buffers": "same A/B/output allocations for both variants throughout timing",
            "cache": "steady repeated buffers; no explicit flush; not a guarantee of full cache residency",
            "clock_policy": "default dynamic clocks and power; not modified",
            "stream": "torch default stream with tvm_ffi.use_torch_stream",
            "cuda_graphs": False, "ncu_during_timing": False,
            "excludes": ["loading", "compilation", "allocation", "input generation", "reference", "validation", "warmup"],
            "geometry": "148 CTAs, 74 two-CTA clusters, 148 logical output tiles, two logical task rounds",
            "scope": "one baseline-versus-hoist comparison at the existing best shape; no cuBLAS or shape sweep",
        },
    }
    start_result(path, result)
    try:
        import torch
        import tvm
        import tvm_ffi
        import tvm.support.nvcc  # Register the original runtime CUDA compilation support.

        result.update(configure(torch))
        libraries = {"baseline": BASELINE_PATH, "hoisted": run_dir / "build" / "hoisted.so"}
        result["libraries"] = {variant: library_info(library) for variant, library in libraries.items()}
        if result["libraries"]["baseline"]["sha256"] != BASELINE_SHA256:
            raise RuntimeError("Baseline artifact differs from the previously measured wide binary")
        if result["libraries"]["hoisted"]["sha256"] == BASELINE_SHA256:
            raise RuntimeError("Candidate binary is identical to the baseline artifact")
        modules, kernels = {}, {}
        for variant, library in libraries.items():
            modules[variant], kernels[variant] = load_kernel(tvm, library)
        m, n, k = SHAPE
        a = torch.empty((m, k), device="cuda", dtype=torch.float16)
        b = torch.empty((n, k), device="cuda", dtype=torch.float16)
        d = torch.empty((m, n), device="cuda", dtype=torch.float16)
        result["shared_buffer_addresses"] = {"a": a.data_ptr(), "b": b.data_ptr(), "d": d.data_ptr()}
        calls = {variant: (lambda kernel=kernel: kernel(a, b, d)) for variant, kernel in kernels.items()}
        result["status"] = "validating"
        write_json(path, result)
        with torch.inference_mode(), tvm_ffi.use_torch_stream():
            for seed in SEEDS:
                torch.manual_seed(seed)
                a.normal_()
                b.normal_()
                reference = torch.mm(a.float(), b.float().T).half()
                for variant in VARIANTS:
                    d.fill_(float("nan"))
                    calls[variant]()
                    torch.cuda.synchronize()
                    check = {"variant": variant, **validate(torch, d, reference, seed)}
                    result["validation"]["before"].append(check)
                    write_json(path, result)
                    print("VALIDATION_BEFORE", json.dumps(check), flush=True)
                    if not check["passed"]:
                        raise RuntimeError(f"Correctness failure: {variant}, seed {seed}")
                    if variant == "baseline":
                        baseline_output = d.clone()
                    else:
                        direct = {
                            "seed": seed, "checked_elements": d.numel(),
                            "bitwise_mismatch_count": int(torch.count_nonzero(
                                d.view(torch.int16) != baseline_output.view(torch.int16)).item()),
                            "max_abs_difference": float((d.float() - baseline_output.float()).abs().max().item()),
                        }
                        result["validation"]["direct_comparison"].append(direct)
                        write_json(path, result)
                        print("DIRECT_COMPARISON", json.dumps(direct), flush=True)
                        del baseline_output
            patterns = ["ABBA"] * (BLOCKS // 2) + ["BAAB"] * (BLOCKS // 2)
            random.Random(ORDER_SEED).shuffle(patterns)
            result["planned_patterns"] = patterns
            result["status"] = "timing"
            write_json(path, result)
            for block_index, pattern in enumerate(patterns):
                block = {"block_index": block_index, "pattern": pattern,
                         "started_at": utc_now(), "slots": []}
                for position, letter in enumerate(pattern):
                    variant = "baseline" if letter == "A" else "hoisted"
                    measurement = timed_slot(torch, calls[variant])
                    block["slots"].append({"position": position + 1, "variant": variant, **measurement})
                means = {variant: statistics.fmean(
                    slot["mean_ms"] for slot in block["slots"] if slot["variant"] == variant)
                    for variant in VARIANTS}
                block["paired"] = {"baseline_mean_ms": means["baseline"],
                                   "hoisted_mean_ms": means["hoisted"],
                                   "baseline_over_hoisted": means["baseline"] / means["hoisted"]}
                block["finished_at"] = utc_now()
                block["telemetry_after"] = telemetry()
                result["blocks"].append(block)
                result["summary"] = summarize(result["blocks"])
                write_json(path, result)
                print("TIMING_BLOCK", json.dumps({"index": block_index, "pattern": pattern,
                                                   **block["paired"]}), flush=True)
            result["status"] = "post_validating"
            write_json(path, result)
            for variant in VARIANTS:
                d.fill_(float("nan"))
                calls[variant]()
                torch.cuda.synchronize()
                check = {"variant": variant, **validate(torch, d, reference, SEEDS[-1])}
                result["validation"]["after"].append(check)
                write_json(path, result)
                print("VALIDATION_AFTER", json.dumps(check), flush=True)
                if not check["passed"]:
                    raise RuntimeError(f"Post-timing correctness failure: {variant}")
        result.update(status="complete", finished_at=utc_now(), final_telemetry=telemetry())
        write_json(path, result)
        print("BENCHMARK_SUMMARY", json.dumps(result["summary"]), flush=True)
        return result
    except BaseException:
        result.update(status="failed", failed_at=utc_now(), error=traceback.format_exc())
        write_json(path, result)
        raise


def profile_candidate(run_dir):
    path = run_dir / "profile_candidate" / "profile_target.json"
    result = {
        "status": "starting", "started_at": utc_now(), "shape": list(SHAPE),
        "variant": "hoisted", "arch": "sm_103a",
        "protocol": {"seed": 2, "warmup_calls": WARMUP_CALLS, "nvtx_range": NVTX_RANGE,
                     "profiled_application_invocations": 1, "cache_control": "no explicit flush",
                     "stream": "torch default stream with tvm_ffi.use_torch_stream",
                     "clock_policy": "unchanged", "timer": None,
                     "reference": "FP32 torch.mm with TF32 disabled, rounded to FP16",
                     "ncu_replay": "NCU may replay this single selected invocation for counters"},
    }
    start_result(path, result)
    try:
        import torch
        import tvm
        import tvm_ffi
        import tvm.support.nvcc

        result.update(configure(torch))
        library = run_dir / "build" / "hoisted.so"
        result["library"] = library_info(library)
        write_json(path, result)
        module, kernel = load_kernel(tvm, library)
        m, n, k = SHAPE
        with torch.inference_mode(), tvm_ffi.use_torch_stream():
            torch.manual_seed(2)
            a = torch.empty((m, k), device="cuda", dtype=torch.float16).normal_()
            b = torch.empty((n, k), device="cuda", dtype=torch.float16).normal_()
            d = torch.full((m, n), float("nan"), device="cuda", dtype=torch.float16)
            for _ in range(WARMUP_CALLS):
                kernel(a, b, d)
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_push(NVTX_RANGE)
            try:
                kernel(a, b, d)
            finally:
                torch.cuda.nvtx.range_pop()
            torch.cuda.synchronize()
            reference = torch.mm(a.float(), b.float().T).half()
            result["validation"] = validate(torch, d, reference, 2)
        if not result["validation"]["passed"]:
            raise RuntimeError("Profiled candidate failed full-output validation")
        result.update(status="complete", finished_at=utc_now(), final_telemetry=telemetry())
        write_json(path, result)
        print("PROFILE_TARGET", json.dumps(result["validation"]), flush=True)
        return result
    except BaseException:
        result.update(status="failed", failed_at=utc_now(), error=traceback.format_exc())
        write_json(path, result)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--profile-variant", choices=["hoisted"])
    args = parser.parse_args()
    run_dir = args.run_dir.resolve(strict=True)
    if args.profile_variant:
        profile_candidate(run_dir)
    else:
        benchmark(run_dir)


if __name__ == "__main__":
    main()
