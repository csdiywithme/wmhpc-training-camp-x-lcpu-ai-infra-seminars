#!/usr/bin/env python3
"""Build an isolated full-forward module; no GPU is needed when --arch is set.

Example: python build.py --flash-root /opt/FlashKDA --arch 100a \
  --output /tmp/c1-nextgen-fused --variant fused
"""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flash-root", type=Path, default=Path("/opt/FlashKDA"))
    parser.add_argument("--arch", choices=["100a", "103a"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=["fused", "split-qk", "split-final", "split-both", "baseline",
                                             "direct", "direct-rowmajor", "direct-release", "direct-wide"], default="fused")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--generate-only", action="store_true", help="Write/audit generated sources without CUDA compilation")
    args = parser.parse_args()
    root = args.flash_root.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parent
    os.environ["MAX_JOBS"] = str(args.jobs)
    # Explicit gencode below avoids PyTorch querying a CUDA device on build workers.
    os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0" if args.arch == "100a" else "10.3"
    module_name = f"c1_nextgen_{args.variant.replace('-', '_')}_{args.arch}"
    direct=args.variant in ["direct","direct-rowmajor","direct-release","direct-wide"]
    kernel_file="k2_tcgen05_direct_wide.cuh" if args.variant=="direct-wide" else "k2_tcgen05_direct.cuh" if direct else "k2_tcgen05.cuh"
    primitive_audit=None
    if args.variant=="direct-wide":
        primitive_source=root/"cutlass/include/cute/arch/copy_sm100.hpp"
        primitive_text=primitive_source.read_text()
        match=re.search(r"struct\s+SM100_TMEM_LOAD_32dp32b16x\s*\{",primitive_text)
        if not match:
            raise RuntimeError("Pinned CUTLASS lacks SM100_TMEM_LOAD_32dp32b16x; refusing guessed primitive")
        next_struct=re.search(r"\nstruct\s+",primitive_text[match.end():])
        end=match.end()+next_struct.start() if next_struct else len(primitive_text)
        snippet=primitive_text[match.start():end]
        if not re.search(r"DRegisters\s*=\s*uint32_t\s*\[\s*16\s*\]",snippet):
            raise RuntimeError("Pinned 16x primitive does not expose expected 16 uint32 destination registers")
        if "tcgen05.ld.sync.aligned.32x32b.x16.b32" not in snippet:
            raise RuntimeError("Pinned primitive has unexpected TMEM instruction; audit required")
        (out/"pinned-tmem-load-primitive.txt").write_text(snippet)
        primitive_audit={"source":str(primitive_source),"source_sha256":sha(primitive_source),
                         "struct":"SM100_TMEM_LOAD_32dp32b16x","destination_registers":16,
                         "instruction":"tcgen05.ld.sync.aligned.32x32b.x16.b32"}
    cpp = (root / "csrc/flash_kda.cpp").read_text()
    cpp = cpp.replace('PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {',
                      'void c1_set_k1_enabled(bool enabled);\n'
                      'PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n'
                      '    m.def("set_k1_enabled", &c1_set_k1_enabled);')
    launcher = (root / "csrc/smxx/fwd_launch.cu").read_text()
    marker = "#if BLOCK_LEVEL_K1 >= 0\n    {"
    if launcher.count(marker) != 1:
        raise RuntimeError("Unexpected upstream K1 launcher; refuse heuristic patch")
    launcher = launcher.replace(marker, "#if BLOCK_LEVEL_K1 >= 0\n    if (c1_k1_enabled) {")
    globals_text = ("static bool c1_k1_enabled = true;\n"
                    "void c1_set_k1_enabled(bool enabled) { c1_k1_enabled = enabled; }\n")
    if args.variant != "baseline":
        start = launcher.index("#if BLOCK_LEVEL_K2 >= 0")
        end = launcher.index("#endif", start) + len("#endif")
        replacement = """c1_nextgen::launch(v_ptr, beta_ptr, initial_state_ptr, final_state_ptr,
        out_ptr, ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk,
        ws_tile_prefix, cu_seqlens_ptr, T_total, H, N, total_tiles,
        HasStateIn, HasStateOut, StateFP32, IsVarlen, stream);"""
        launcher = launcher[:start] + replacement + launcher[end:]
        globals_text = f'#include "{kernel_file}"\n' + globals_text
    launcher = globals_text + launcher
    (out / "binding.cpp").write_text(cpp)
    (out / "launcher.cu").write_text(launcher)
    shutil.copy2(here / kernel_file, out / kernel_file)
    fq = int(args.variant not in ["split-qk", "split-both"])
    ff = int(args.variant not in ["split-final", "split-both"])
    nvcc_flags = ["-O3", "-std=c++17", "--expt-relaxed-constexpr", "--expt-extended-lambda",
                  "--use_fast_math", "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
                  "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                  "--ptxas-options=-v,--register-usage-level=10,--warn-on-spills", "-lineinfo", "--threads=4",
                  f"-gencode=arch=compute_{args.arch},code=sm_{args.arch}",
                  f"-DC1_FUSE_QK={fq}", f"-DC1_FUSE_FINAL={ff}",
                  f"-DC1_TRANSPOSE_SHARED={int(args.variant in ['direct','direct-release','direct-wide'])}"]
    if args.variant in ["direct-release","direct-wide"]:
        nvcc_flags.append("-DNDEBUG")
    metadata = {"module": module_name, "variant": args.variant, "arch": args.arch,
                "started_unix": time.time(), "argv": sys.argv,
                "k2_source_sha256": sha(here / kernel_file), "k2_source_file":kernel_file,
                "generated_launcher_sha256": sha(out / "launcher.cu"),
                "upstream_launcher_sha256": sha(root / "csrc/smxx/fwd_launch.cu"),
                "nvcc_flags": nvcc_flags, "status": "building",
                "tmem_primitive_audit":primitive_audit}
    for repo, path in [("flash", root), ("cutlass", root / "cutlass")]:
        if path.exists():
            proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True)
            metadata[f"{repo}_commit"] = proc.stdout.strip() if proc.returncode == 0 else None
        else:
            metadata[f"{repo}_commit"] = None
    manifest = out / "manifest.json"
    manifest.write_text(json.dumps(metadata, indent=2))
    if args.generate_only:
        metadata.update(status="generated_only",finished_unix=time.time())
        manifest.write_text(json.dumps(metadata,indent=2))
        print(json.dumps(metadata,indent=2))
        return
    try:
        from torch.utils.cpp_extension import load
        module = load(name=module_name, sources=[str(out / "binding.cpp"), str(out / "launcher.cu")],
                      extra_cflags=["-O3", "-std=c++17", "-Wno-psabi"], extra_cuda_cflags=nvcc_flags,
                      extra_include_paths=[str(root / "csrc"), str(root / "csrc/smxx"),
                                           str(root / "cutlass/include"), str(root / "cutlass/tools/util/include"),
                                           str(root / "cutlass/examples/common"), str(out)],
                      build_directory=str(out), verbose=True)
        metadata.update(status="built", module_path=Path(module.__file__).name)
    except Exception as exc:
        metadata.update(status="failed", error=repr(exc))
        raise
    finally:
        metadata["finished_unix"] = time.time()
        manifest.write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
