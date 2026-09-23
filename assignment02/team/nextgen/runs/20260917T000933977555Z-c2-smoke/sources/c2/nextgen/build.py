"""CPU-only CUDA compilation: python nextgen/build.py --arch 103a --output DIR.

Requires CUDA 13.1, Torch 2.10 CUDA build, ninja, a C++17 compiler, and CUTLASS
headers. No CUDA device enumeration, allocation, or kernel execution occurs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="103a", help="Comma-separated 100a/103a")
    parser.add_argument("--variant", choices=("original", "coalesced", "coalesced_pad17", "coalesced_release", "coalesced_wide", "original_wide"), default="original")
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
    variant_id = {"original": 0, "coalesced": 1, "coalesced_pad17": 2,
                  "coalesced_release": 3, "coalesced_wide": 4, "original_wide": 5}[args.variant]
    flags.append(f"-DC2_DATAFLOW_VARIANT={variant_id}")
    if variant_id in (3, 4, 5):
        flags.append("-DNDEBUG")
    primitive_audit = None
    if variant_id in (4, 5):
        primitive_source = args.cutlass / "include/cute/arch/copy_sm100.hpp"
        primitive_bytes = primitive_source.read_bytes()
        primitive_text = primitive_bytes.decode()
        match = re.search(r"struct\s+SM100_TMEM_LOAD_32dp32b16x\s*\{", primitive_text)
        if not match:
            raise RuntimeError("Pinned CUTLASS lacks SM100_TMEM_LOAD_32dp32b16x; refusing a guessed primitive")
        next_struct = re.search(r"\nstruct\s+", primitive_text[match.end():])
        end = match.end() + next_struct.start() if next_struct else len(primitive_text)
        snippet = primitive_text[match.start():end]
        if not re.search(r"DRegisters\s*=\s*uint32_t\s*\[\s*16\s*\]", snippet):
            raise RuntimeError("Pinned 16x primitive lacks the expected 16 uint32 destination registers")
        if "tcgen05.ld.sync.aligned.32x32b.x16.b32" not in snippet:
            raise RuntimeError("Pinned 16x primitive instruction differs; audit required")
        (args.output / "pinned-tmem-load-primitive.txt").write_text(snippet)
        primitive_audit = {"source": str(primitive_source),
                           "source_sha256": hashlib.sha256(primitive_bytes).hexdigest(),
                           "struct": "SM100_TMEM_LOAD_32dp32b16x", "destination_registers": 16,
                           "instruction": "tcgen05.ld.sync.aligned.32x32b.x16.b32"}
    import torch
    from torch.utils.cpp_extension import load
    started = time.monotonic()
    record = {"source_hashes": hashes, "architectures": archs, "cuda_flags": flags,
              "variant": args.variant, "variant_id": variant_id,
              "ndebug": variant_id in (3, 4, 5), "tmem_elements_per_lane": 16 if variant_id in (4, 5) else 1,
              "tmem_primitive_audit": primitive_audit,
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "cutlass": str(args.cutlass), "scope": "CPU compile only"}
    try:
        module = load(name="c2_nextgen_ext", sources=[str(p) for p in files],
                      extra_include_paths=[str(args.cutlass / "include"), str(args.cutlass / "tools/util/include")],
                      extra_cflags=["-O3", "-std=c++17"], extra_cuda_cflags=flags,
                      build_directory=str(args.output), verbose=True)
        record["inverse_layout_coordinate_mismatches"] = module.layout_map_check()
        if record["inverse_layout_coordinate_mismatches"] != 0:
            raise RuntimeError("Logical-to-partition inverse failed the exhaustive CPU coordinate check")
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
