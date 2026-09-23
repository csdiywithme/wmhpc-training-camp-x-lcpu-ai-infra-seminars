"""Compile one GEMM version without allocating GPU memory or launching a kernel.

Use a fresh process for every version, as recommended by the upstream tutorial.
The resulting shared library is loaded on the GPU worker with
``tvm.runtime.load_module(path).get_function("main", query_imports=True)``.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import platform
import time
import traceback


UPSTREAM_REVISION = "ebccca2e5675966f68fb3d4880d4448194bd638d"


def _write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cuda_source(module):
    """Find CUDA source in the exported host module's import tree."""
    errors = []
    pending = list(module.imports)
    while pending:
        child = pending.pop(0)
        pending.extend(child.imports)
        try:
            source = child.inspect_source("cuda")
            if source:
                return source
        except Exception as exc:
            errors.append(str(exc))
        try:
            source = child.inspect_source()
            if source and ("__global__" in source or "#include <cuda" in source):
                return source
        except Exception as exc:
            errors.append(str(exc))
    raise RuntimeError("No generated CUDA source found: " + "; ".join(errors))


def compile_one(version, size, sm_count, output, arch="sm_103a"):
    """Return compilation metadata and save all compiler artifacts to output."""
    if version not in range(1, 10):
        raise ValueError("version must be between 1 and 9")
    if size <= 0 or size % 128:
        raise ValueError("size must be a positive multiple of 128")
    if version in (6, 8) and size % 256:
        raise ValueError("versions 6 and 8 require size divisible by 256")
    if version == 9 and size % 512:
        raise ValueError("version 9 requires size divisible by 512")
    if sm_count <= 0 or (version in (8, 9) and sm_count % 2):
        raise ValueError("sm_count must be positive and even for two-CTA kernels")

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    stem = f"v{version}"
    paths = {
        "library": output / f"{stem}.so",
        "tirx": output / f"{stem}.tirx.py",
        "cuda": output / f"{stem}.cu",
        "metadata": output / f"{stem}.compile.json",
    }
    metadata = {
        "status": "started",
        "version": version,
        "shape": [size, size, size],
        "dtype": "float16",
        "accumulator_dtype": "float32",
        "operation": "D = A @ B.T",
        "arch": arch,
        "sm_count": sm_count,
        "upstream_revision": UPSTREAM_REVISION,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "files": {key: path.name for key, path in paths.items()},
        "entrypoint": "main",
        "gpu_launched": False,
    }
    _write_json(paths["metadata"], metadata)
    overall_start = time.perf_counter()
    try:
        import tvm

        metadata["tvm_version"] = tvm.__version__
        module_name = "kernels_basics" if version <= 3 else "kernels_advanced"
        factories = importlib.import_module(module_name)
        metadata["factory_source_sha256"] = _sha256(Path(factories.__file__))
        factory = getattr(factories, f"hgemm_v{version}")
        factory_start = time.perf_counter()
        if version >= 6:
            kernel = factory(size, size, size, sm_count=sm_count)
        else:
            kernel = factory(size, size, size)
        metadata["factory_seconds"] = time.perf_counter() - factory_start
        paths["tirx"].write_text(kernel.script() + "\n")
        target = tvm.target.Target({"kind": "cuda", "arch": arch}).with_host("llvm")
        metadata["target"] = str(target)
        metadata["host_target"] = str(target.host)
        compile_start = time.perf_counter()
        with target:
            executable = tvm.compile(
                tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx"
            )
        metadata["compile_seconds"] = time.perf_counter() - compile_start
        paths["cuda"].write_text(_cuda_source(executable.mod))
        export_start = time.perf_counter()
        executable.export_library(str(paths["library"]))
        metadata["export_seconds"] = time.perf_counter() - export_start
        metadata["sha256"] = {
            key: _sha256(paths[key]) for key in ("library", "tirx", "cuda")
        }
        metadata["status"] = "compiled"
        return metadata
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["error"] = str(exc)
        metadata["traceback"] = traceback.format_exc()
        raise
    finally:
        metadata["total_seconds"] = time.perf_counter() - overall_start
        metadata["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(paths["metadata"], metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", type=int, choices=range(1, 10), required=True)
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--sm-count", type=int, default=148)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arch", default="sm_103a")
    args = parser.parse_args()
    metadata = compile_one(
        args.version, args.size, args.sm_count, args.output, args.arch
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
