"""Step 2: pinned FlashKDA install and small correctness check on one B300."""
import json
from datetime import datetime, timezone
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
image = (
    modal.Image.from_registry("nvidia/cuda:13.1.0-devel-ubuntu24.04", add_python="3.11")
    .entrypoint([])
    .apt_install("git", "build-essential", "ninja-build")
    .pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("setuptools==80.9.0", "wheel==0.45.1", "ninja==1.11.1.4")
    .env({"FLASH_KDA_CUDA_ARCHS": "103a", "TORCH_CUDA_ARCH_LIST": "10.3a",
          "MAX_JOBS": "2", "NVCC_THREADS": "2",
          # Modal's bundled Python may retain clang linker defaults; use build-essential.
          "CC": "gcc", "CXX": "g++", "LDSHARED": "gcc -shared"})
    .run_commands(
        "gcc --version && g++ --version",
        "git clone https://github.com/MoonshotAI/FlashKDA.git /opt/FlashKDA",
        "git -C /opt/FlashKDA checkout --detach 1ce47ea",
        "git -C /opt/FlashKDA submodule update --init --recursive",
        "python -c \"import subprocess; rev=subprocess.check_output(['git','-C','/opt/FlashKDA/cutlass','rev-parse','HEAD'],text=True).strip(); assert rev.startswith('5c149f5'), rev\"",
        "python -m pip install -v --no-build-isolation --no-deps /opt/FlashKDA",
    )
    .add_local_file(HERE / "smoke_flashkda.py", "/opt/smoke_flashkda.py")
    .add_local_file(HERE / "bench_flashkda.py", "/opt/bench_flashkda.py")
)
app = modal.App("c1-flashkda-smoke")


@app.function(image=image, gpu="B300", cpu=2, memory=8192, timeout=300,
              retries=0, max_containers=1, scaledown_window=2)
def smoke(mode="smoke"):
    import subprocess
    scripts = {"smoke": "/opt/smoke_flashkda.py", "bench": "/opt/bench_flashkda.py"}
    script = scripts[mode]
    results = []
    for cmd, limit in [
        (["nvidia-smi"], 15),
        (["git", "-C", "/opt/FlashKDA", "rev-parse", "HEAD"], 10),
        (["git", "-C", "/opt/FlashKDA", "submodule", "status"], 10),
        (["python", "-m", "pip", "freeze"], 15),
        (["python", "-u", script], 240),
        (["nvidia-smi"], 15),
    ]:
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=limit)
            record = dict(command=cmd, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
        except subprocess.TimeoutExpired as exc:
            def as_text(value):
                return value.decode(errors="replace") if isinstance(value, bytes) else value
            record = dict(command=cmd, returncode=None, error="timeout",
                          stdout=as_text(exc.stdout), stderr=as_text(exc.stderr))
        results.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    return results


@app.local_entrypoint()
def main(mode: str = "smoke"):
    if mode not in ("smoke", "bench"):
        raise ValueError("mode must be smoke or bench")
    results = smoke.remote(mode)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = HERE / "results" / f"flashkda-{mode}-{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(f"Saved: {output}")
    if any(r["returncode"] != 0 for r in results):
        raise RuntimeError(f"{mode} step failed; inspect saved output before rerunning.")
