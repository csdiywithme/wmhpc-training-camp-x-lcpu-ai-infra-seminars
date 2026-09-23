"""Pinned, single-GPU C1 experiments. All remote artifacts are returned locally."""
import json
from datetime import datetime, timezone
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
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
image = (
    base_image.add_local_file(HERE / "run_experiments.py", "/opt/run_experiments.py")
    .add_local_file(HERE / "fla_kda_ref/naive.py", "/opt/c1_naive.py")
    .add_local_file(HERE / "tile_microbench.cu", "/opt/tile_microbench.cu")
    .add_local_file(HERE / "chunk_numeric_gpu.py", "/opt/chunk_numeric_gpu.py")
)
challenge_image = (
    base_image.add_local_file(HERE / "challenge_build.py", "/opt/challenge_build.py", copy=True)
    .run_commands("python /opt/challenge_build.py --build")
    .add_local_file(HERE / "run_experiments.py", "/opt/run_experiments.py")
    .add_local_file(HERE / "fla_kda_ref/naive.py", "/opt/c1_naive.py")
)
app = modal.App("c1-flashkda-experiments")


@app.function(image=image, gpu="B300", cpu=4, memory=16384, timeout=900,
              retries=0, max_containers=1, scaledown_window=2)
def run(mode: str, heads: int):
    import subprocess
    import traceback
    records = []
    for cmd, limit in [
        (["nvidia-smi"], 20),
        (["git", "-C", "/opt/fla", "rev-parse", "HEAD"], 10),
        (["python", "-m", "pip", "freeze"], 20),
        (["python", "-u", "/opt/run_experiments.py", "--mode", mode, "--heads", str(heads)], 800),
    ]:
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=limit)
            record = dict(command=cmd, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
        except Exception:
            record = dict(command=cmd, returncode=None, error=traceback.format_exc())
        records.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    artifacts = {}
    for path in Path("/tmp/c1-artifacts").glob("*"):
        if path.is_file():
            artifacts[path.name] = path.read_bytes()
    return records, artifacts


@app.function(image=challenge_image, gpu="B300", cpu=4, memory=16384, timeout=900,
              retries=0, max_containers=1, scaledown_window=2)
def run_challenge(heads: int, mode: str = "challenge"):
    records, artifacts = run.local(mode, heads)
    artifacts["challenge.patch"] = Path("/opt/c1-value-split/challenge.patch").read_bytes()
    return records, artifacts


@app.function(image=base_image, cpu=2, memory=2048, timeout=120, retries=0,
              max_containers=1, scaledown_window=2)
def export_report(data: bytes):
    import subprocess
    Path("/tmp/input.ncu-rep").write_bytes(data)
    p = subprocess.run(["ncu", "--import", "/tmp/input.ncu-rep", "--page", "raw", "--csv"],
                       capture_output=True, text=True, timeout=100)
    return dict(returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)


@app.local_entrypoint()
def main(mode: str = "official", heads: int = 96, report: str = ""):
    if mode == "export":
        path = Path(report)
        result = export_report.remote(path.read_bytes())
        path.with_suffix(".csv").write_text(result["stdout"])
        path.with_suffix(".export.json").write_text(json.dumps(result, indent=2))
        assert result["returncode"] == 0
        print(f"Saved: {path.with_suffix('.csv')}")
        return
    assert mode in ("official", "profile", "precision", "long_precision", "challenge", "challenge_profile", "tile", "numeric")
    records, artifacts = run_challenge.remote(heads, mode) if mode.startswith("challenge") else run.remote(mode, heads)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = HERE / "results" / f"c1-{mode}-h{heads}-{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    (out / "commands.json").write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n")
    for name, data in artifacts.items():
        (out / name).write_bytes(data)
    print(f"Saved: {out}")
    if any(r["returncode"] != 0 for r in records):
        raise RuntimeError("Experiment subprocess failed; raw output has been saved")
