"""CPU-only CUDA compilation: python nextgen/build.py --arch 103a --output DIR.

Requires CUDA 13.1, Torch 2.10 CUDA build, ninja, a C++17 compiler, and CUTLASS
headers. No CUDA device enumeration, allocation, or kernel execution occurs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="103a", help="Comma-separated 100a/103a")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cutlass", type=Path, default=Path(os.environ.get("CUTLASS_PATH", "/opt/FlashKDA/cutlass")))
    args = parser.parse_args()
    archs = args.arch.split(",")
    if not archs or any(a not in ("100a", "103a") for a in archs):
        parser.error("Expected 100a and/or 103a")
    if not (args.cutlass / "include/cute/tensor.hpp").is_file():
        parser.error("CUTLASS headers not found")
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parent
    files = [root / "binding.cpp", root / "partial.cu"]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    os.environ.setdefault("MAX_JOBS", "2")
    # Explicit arch flags prevent cpp_extension from probing visible GPUs.
    flags = ["-O3", "-std=c++17", "-lineinfo", "--expt-relaxed-constexpr",
             "--expt-extended-lambda", "--ptxas-options=-v"]
    flags += [f"-gencode=arch=compute_{a},code=sm_{a}" for a in archs]
    import torch
    from torch.utils.cpp_extension import load
    started = time.monotonic()
    record = {"source_hashes": hashes, "architectures": archs, "cuda_flags": flags,
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "cutlass": str(args.cutlass), "scope": "CPU compile only"}
    try:
        module = load(name="c2_nextgen_ext", sources=[str(p) for p in files],
                      extra_include_paths=[str(args.cutlass / "include"), str(args.cutlass / "tools/util/include")],
                      extra_cflags=["-O3", "-std=c++17"], extra_cuda_cflags=flags,
                      build_directory=str(args.output), verbose=True)
        record.update(status="COMPILED", module=str(Path(module.__file__).resolve()))
        for p in files:
            shutil.copy2(p, args.output / (p.name + ".source"))
        tool = shutil.which("cuobjdump")
        if tool:
            dumped = subprocess.run([tool, "--dump-sass", module.__file__], capture_output=True, text=True)
            (args.output / "partial.sass").write_text(dumped.stdout)
            record["sass_returncode"] = dumped.returncode
            record["sass_has_tcgen05"] = "UTC" in dumped.stdout
    except Exception as error:
        record.update(status="BUILD_FAILED", error=repr(error))
        raise
    finally:
        record["seconds"] = time.monotonic() - started
        (args.output / "build.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
