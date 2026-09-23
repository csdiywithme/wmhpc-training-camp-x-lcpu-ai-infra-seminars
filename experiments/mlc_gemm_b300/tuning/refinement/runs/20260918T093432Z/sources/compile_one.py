"""Compile one rectangular workload/candidate in a fresh CPU process."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import re
import subprocess
import time
import traceback

def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")

def compile_one(case, variant, output, sm_count=148):
    import tvm
    m, n, k = case["shape"]
    if min(m, n, k) <= 0 or m % 512 or n % 256 or k % 64:
        raise ValueError("Expected M%512=N%256=K%64=0")
    if not re.fullmatch(r"[a-z0-9_]+", case["name"]):
        raise ValueError("Invalid case name")
    output.mkdir(parents=True, exist_ok=True)
    stem = case["name"] + "__" + variant
    meta_path = output / (stem + ".compile.json")
    metadata = {"case": case, "variant": variant, "arch": "sm_103a", "sm_count": sm_count,
                "status": "started", "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "tvm": tvm.__version__, "library_cuda_compilation": "at GPU module load, outside timing"}
    write(meta_path, metadata)
    started = time.monotonic()
    try:
        if variant in ("release", "k128"):
            module = "kernels_release" if variant == "release" else "kernels_k128"
            factories = importlib.import_module(module)
            factory = factories.hgemm_v9_release if variant == "release" else factories.hgemm_v9_k128
            kernel = factory(m, n, k, sm_count=sm_count)
        elif variant == "original":
            factories = importlib.import_module("kernels_advanced")
            kernel = factories.hgemm_v9(m, n, k, sm_count=sm_count)
        else:
            factories = importlib.import_module("kernels_tuned")
            if variant == "fenced":
                kernel = factories.hgemm_v9_fenced(m, n, k, sm_count=sm_count)
            elif variant == "wide":
                kernel = factories.hgemm_v9_wide(m, n, k, sm_count=sm_count)
            elif variant in ("full_late", "full_early"):
                kernel = factories.hgemm_v9_full_epilogue(
                    m, n, k, sm_count=sm_count, early_release=(variant == "full_early"))
            else:
                raise ValueError(variant)
        metadata["factory_sha256"] = hashlib.sha256(Path(factories.__file__).read_bytes()).hexdigest()
        (output / (stem + ".tirx.py")).write_text(kernel.script() + "\n")
        target = tvm.target.Target({"kind": "cuda", "arch": "sm_103a"}).with_host("llvm")
        with target:
            executable = tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")
        source = executable.mod.imports[0].inspect_source("cuda")
        cuda = output / (stem + ".cu")
        cuda.write_text(source)
        cubin = output / (stem + ".cubin")
        cmd = ["nvcc", "--cubin", "-O3", "-arch=sm_103a", "--std=c++17",
               "--expt-relaxed-constexpr", "--expt-extended-lambda", "--use_fast_math",
               "--ptxas-options=-v,--register-usage-level=10", str(cuda), "-o", str(cubin)]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        metadata["nvcc"] = {"command": cmd, "returncode": p.returncode,
                            "stdout": p.stdout, "stderr": p.stderr}
        if p.returncode:
            raise RuntimeError("NVCC preflight failed: " + p.stderr)
        lib = output / (stem + ".so")
        executable.export_library(str(lib))
        metadata["sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (cuda, cubin, lib)}
        metadata["status"] = "compiled"
    except BaseException:
        metadata["status"] = "failed"
        metadata["error"] = traceback.format_exc()
        raise
    finally:
        metadata["seconds"] = time.monotonic() - started
        write(meta_path, metadata)
    print(json.dumps(metadata), flush=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--request", type=Path, required=True)
    p.add_argument("--case", required=True)
    p.add_argument("--variant", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    req = json.loads(a.request.read_text())
    case = next(c for c in req["cases"] if c["name"] == a.case)
    compile_one(case, a.variant, a.output, req["sm_count"])
