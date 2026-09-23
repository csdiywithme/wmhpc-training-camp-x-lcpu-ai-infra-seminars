"""Profile one invocation of the saved wide kernel; never rebuild or benchmark it."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path


SHAPE = (2048, 9472, 8192)
RANGE = "wide_2048x9472x8192"
EXPECTED_PACKAGES = {
    "torch": "2.10.0+cu130",
    "apache-tvm": "0.26.0",
    "apache-tvm-ffi": "0.1.14.post0",
    "nvidia-cublas": "13.1.0.3",
    "nvidia-cuda-runtime": "13.0.96",
}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validate(torch, output, reference):
    """Use the benchmark's full-output row-chunked tolerance and error metrics."""
    max_abs, square_sum, reference_square_sum = 0.0, 0.0, 0.0
    mismatch, nonfinite = 0, 0
    for start in range(0, output.shape[0], 1024):
        actual = output[start:start + 1024].float()
        target = reference[start:start + 1024].float()
        error = (actual - target).abs()
        finite = torch.isfinite(actual)
        valid = finite & (error <= 0.01 + 0.02 * target.abs())
        nonfinite += int((~finite).sum().item())
        mismatch += int((~valid).sum().item())
        maximum = float(error.max().item())
        if math.isfinite(maximum):
            max_abs = max(max_abs, maximum)
        value = float(error.square().sum(dtype=torch.float64).item())
        square_sum += value if math.isfinite(value) else 0.0
        reference_square_sum += float(target.square().sum(dtype=torch.float64).item())
    return {
        "seed": 2, "passed": mismatch == 0, "checked_elements": output.numel(),
        "atol": 0.01, "rtol": 0.02, "row_chunk": 1024,
        "mismatch_count": mismatch, "nonfinite_count": nonfinite,
        "max_abs_error": max_abs if nonfinite == 0 else None,
        "rms_error": math.sqrt(square_sum / output.numel()) if nonfinite == 0 else None,
        "relative_l2_error": math.sqrt(square_sum / reference_square_sum)
        if nonfinite == 0 and reference_square_sum else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arch", choices=["sm_103a"], default="sm_103a")
    args = parser.parse_args()

    import torch
    import tvm
    import tvm_ffi
    import tvm.support.nvcc  # Registers the same CUDA compilation support as the benchmark.

    args.output.mkdir(parents=True, exist_ok=True)
    library = args.library.resolve(strict=True)
    packages = {name: importlib.metadata.version(name) for name in EXPECTED_PACKAGES}
    if packages != EXPECTED_PACKAGES:
        raise RuntimeError(f"Packages differ from the saved benchmark: {packages}")
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

    result = {
        "status": "starting", "started_at": datetime.now(timezone.utc).isoformat(),
        "shape": list(SHAPE), "operation": "D[M,N] = A[M,K] @ B[N,K].T",
        "variant": "hgemm_v9_wide", "arch": args.arch,
        "library": str(library), "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "packages": packages, "torch_cuda": torch.version.cuda,
        "gpu": {"name": props.name, "sm_count": props.multi_processor_count,
                "compute_capability": [props.major, props.minor],
                "total_memory_bytes": props.total_memory, "properties": str(props)},
        "protocol": {
            "seed": 2, "dtype": "FP16 inputs/output; FP32 accumulation",
            "reference": "FP32 torch.mm with TF32 off, rounded to FP16",
            "allow_tf32": False, "allow_fp16_reduced_precision_reduction": False,
            "stream": "default CUDA stream with tvm_ffi.use_torch_stream",
            "warmup_calls": 50, "no_explicit_cache_flush": True,
            "nvtx_range": RANGE, "profiled_application_invocations": 1,
            "ncu_replay": "NCU may replay this one selected invocation to collect counters",
            "timer": None, "clock_policy": "unchanged",
            "nan_prefill": "before warmup; same buffers reused through captured invocation",
        },
    }
    metadata = args.output / "profile_target.json"
    write_json(metadata, result)
    with tvm.target.Target({"kind": "cuda", "arch": args.arch}):
        module = tvm.runtime.load_module(str(library))
    kernel = module.get_function("main", query_imports=True)
    m, n, k = SHAPE
    torch.manual_seed(2)
    a = torch.empty((m, k), device="cuda", dtype=torch.float16).normal_()
    b = torch.empty((n, k), device="cuda", dtype=torch.float16).normal_()
    d = torch.full((m, n), float("nan"), device="cuda", dtype=torch.float16)
    with tvm_ffi.use_torch_stream():
        for _ in range(result["protocol"]["warmup_calls"]):
            kernel(a, b, d)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push(RANGE)
        try:
            kernel(a, b, d)
        finally:
            torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()

    # All reference and validation kernels are outside the selected NVTX range.
    reference = torch.mm(a.float(), b.float().T).half()
    result["validation"] = validate(torch, d, reference)
    result.update(status="complete" if result["validation"]["passed"] else "validation_failed",
                  finished_at=datetime.now(timezone.utc).isoformat())
    write_json(metadata, result)
    print("PROFILE_TARGET", json.dumps(result["validation"]), flush=True)
    if not result["validation"]["passed"]:
        raise RuntimeError("Profiled output failed the saved benchmark's full-output tolerance")


if __name__ == "__main__":
    main()
