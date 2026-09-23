"""CUDA 13.1 compiler and the cuBLAS/PyTorch versions used by the old baseline."""

import modal


# Keep the initial layers identical to the previous successful experiment. The
# final explicit pin protects the cuBLAS baseline from dependency drift.
image = (
    modal.Image.from_registry("nvidia/cuda:13.1.0-devel-ubuntu24.04", add_python="3.11")
    .entrypoint([])
    .apt_install("build-essential", "ninja-build")
    .pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("apache-tvm==0.26.0", "cuda-bindings==13.0.3", "numpy==2.2.6")
    .env({"TVM_CUDA_COMPILE_MODE": "nvcc", "MAX_JOBS": "4"})
    .pip_install("nvidia-cublas==13.1.0.3")
)
