"""Pinned B300 environment, frozen with each experimental run."""
import modal

base_image = (
    modal.Image.from_registry("nvidia/cuda:13.1.0-devel-ubuntu24.04", add_python="3.11")
    .entrypoint([])
    .apt_install("git", "build-essential", "ninja-build")
    .pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("setuptools==80.9.0", "wheel==0.45.1", "ninja==1.11.1.4")
    .env({"FLASH_KDA_CUDA_ARCHS": "103a", "TORCH_CUDA_ARCH_LIST": "10.3a",
          "MAX_JOBS": "2", "NVCC_THREADS": "2", "CC": "gcc", "CXX": "g++", "LDSHARED": "gcc -shared"})
    .run_commands(
        "gcc --version && g++ --version",
        "git clone https://github.com/MoonshotAI/FlashKDA.git /opt/FlashKDA",
        "git -C /opt/FlashKDA checkout --detach 1ce47ea",
        "git -C /opt/FlashKDA submodule update --init --recursive",
        "python -c \"import subprocess; rev=subprocess.check_output(['git','-C','/opt/FlashKDA/cutlass','rev-parse','HEAD'],text=True).strip(); assert rev.startswith('5c149f5'), rev\"",
        "python -m pip install -v --no-build-isolation --no-deps /opt/FlashKDA",
    )
    .apt_install("cuda-nsight-compute-13-1")
    .pip_install("einops", "pytest", "transformers", "packaging")
    .env({"FLA_FLASH_KDA": "0", "TOKENIZERS_PARALLELISM": "false"})
    .run_commands(
        "git clone https://github.com/fla-org/flash-linear-attention.git /opt/fla",
        "git -C /opt/fla checkout --detach a3edffc",
        "python -m pip install --no-build-isolation --no-deps /opt/fla",
    )
)
