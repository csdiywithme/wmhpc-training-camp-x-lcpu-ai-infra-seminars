#!/usr/bin/env python3
"""Build an isolated full-forward module; no GPU is needed when --arch is set.

Example: python build.py --flash-root /opt/FlashKDA --arch 100a \
  --output /tmp/c1-nextgen-fused --variant fused
"""
import argparse
import hashlib
import json
import os
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
    parser.add_argument("--variant", choices=["fused", "split-qk", "split-final", "split-both", "baseline"], default="fused")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    root = args.flash_root.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parent
    os.environ["MAX_JOBS"] = str(args.jobs)
    # Explicit gencode below avoids PyTorch querying a CUDA device on build workers.
    os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0" if args.arch == "100a" else "10.3"
    module_name = f"c1_nextgen_{args.variant.replace('-', '_')}_{args.arch}"
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
        globals_text = '#include "k2_tcgen05.cuh"\n' + globals_text
    launcher = globals_text + launcher
    (out / "binding.cpp").write_text(cpp)
    (out / "launcher.cu").write_text(launcher)
    shutil.copy2(here / "k2_tcgen05.cuh", out / "k2_tcgen05.cuh")
    fq = int(args.variant not in ["split-qk", "split-both"])
    ff = int(args.variant not in ["split-final", "split-both"])
    nvcc_flags = ["-O3", "-std=c++17", "--expt-relaxed-constexpr", "--expt-extended-lambda",
                  "--use_fast_math", "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
                  "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                  "--ptxas-options=-v,--warn-on-spills", "-lineinfo", "--threads=4",
                  f"-gencode=arch=compute_{args.arch},code=sm_{args.arch}",
                  f"-DC1_FUSE_QK={fq}", f"-DC1_FUSE_FINAL={ff}"]
    metadata = {"module": module_name, "variant": args.variant, "arch": args.arch,
                "started_unix": time.time(), "argv": sys.argv,
                "k2_source_sha256": sha(here / "k2_tcgen05.cuh"),
                "generated_launcher_sha256": sha(out / "launcher.cu"),
                "upstream_launcher_sha256": sha(root / "csrc/smxx/fwd_launch.cu"),
                "nvcc_flags": nvcc_flags, "status": "building"}
    for repo, path in [("flash", root), ("cutlass", root / "cutlass")]:
        proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True)
        metadata[f"{repo}_commit"] = proc.stdout.strip() if proc.returncode == 0 else None
    manifest = out / "manifest.json"
    manifest.write_text(json.dumps(metadata, indent=2))
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
