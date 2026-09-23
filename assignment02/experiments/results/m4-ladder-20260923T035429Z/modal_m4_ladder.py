"""Run the A01 naive and A02 M4 ladder in one bounded B300 allocation."""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
ROOT = Path("/opt/m4_ladder")
A02_CUDA = ROOT / "assignment02/cuda"
INPUTS = (
    "assignment01/cuda/common.h",
    "assignment01/cuda/bonus/matmul.cu",
    "assignment02/experiments/m4_naive_fp32_4096.cu",
    "assignment02/cuda/common.h",
    "assignment02/cuda/Makefile",
    "assignment02/cuda/m4_gemm/01_tiled.cu",
    "assignment02/cuda/m4_gemm/02_tma.cu",
    "assignment02/cuda/m4_gemm/03_pipeline.cu",
)
TARGETS = ("01_tiled", "02_tma", "03_pipeline")


def command(argv, cwd=ROOT, timeout=90):
    try:
        proc = subprocess.run(argv, cwd=cwd, capture_output=True, text=True,
                              timeout=timeout)
        record = dict(command=argv, cwd=str(cwd), returncode=proc.returncode,
                      stdout=proc.stdout, stderr=proc.stderr)
    except subprocess.TimeoutExpired:
        record = dict(command=argv, cwd=str(cwd), returncode=None,
                      error="timeout")
    print(json.dumps(record), flush=True)
    return record


def build():
    records = [command([
        "nvcc", "-O2", "-std=c++17", "-I", str(ROOT / "assignment01/cuda"),
        "-gencode", "arch=compute_100f,code=sm_100f",
        "-o", str(ROOT / "naive4096"),
        "assignment02/experiments/m4_naive_fp32_4096.cu",
    ], timeout=120)]
    if records[-1]["returncode"] != 0:
        raise RuntimeError("A01 naive compile failed")
    for target in TARGETS:
        records.append(command([
            "make", "-B", "ARCH=100f", "STAGES=3",
            f"bin/m4_gemm/{target}",
        ], cwd=A02_CUDA, timeout=120))
        if records[-1]["returncode"] != 0:
            raise RuntimeError(f"{target} compile failed")
    (ROOT / "build.json").write_text(json.dumps(records, indent=2))


image = (modal.Image.from_registry("nvidia/cuda:13.1.0-devel-ubuntu24.04",
                                   add_python="3.11")
         .entrypoint([]).apt_install("build-essential"))
for name in INPUTS:
    image = image.add_local_file(REPO / name, str(ROOT / name), copy=True)
image = image.run_function(build, timeout=480)
app = modal.App("assignment02-m4-ladder")


@app.function(image=image, gpu="B300", timeout=60, retries=0,
              max_containers=1, scaledown_window=2)
def run():
    records = [command(["nvidia-smi"], timeout=10)]
    records.append(command(["timeout", "-k", "2s", "5s", "./naive4096"],
                           timeout=10))
    for target in TARGETS:
        records.append(command(["timeout", "-k", "2s", "5s",
                                f"./bin/m4_gemm/{target}", "4096", "4096", "4096"],
                               cwd=A02_CUDA, timeout=10))
    records.append(command(["nvidia-smi", "--query-compute-apps=pid,process_name",
                            "--format=csv"], timeout=10))
    return dict(commands=records,
                build=json.loads((ROOT / "build.json").read_text()),
                input_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                              for name in INPUTS})


@app.local_entrypoint()
def main():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = HERE / "results" / ("m4-ladder-" + stamp)
    out.mkdir(parents=True)
    for name in INPUTS:
        target = out / "inputs" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((REPO / name).read_bytes())
    (out / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    try:
        result = run.remote()
        (out / "run.json").write_text(json.dumps(result, indent=2) + "\n")
        for name in INPUTS:
            digest = hashlib.sha256((out / "inputs" / name).read_bytes()).hexdigest()
            if digest != result["input_sha256"][name]:
                raise RuntimeError("Input snapshot differs from remote source")
        print("Saved:", out)
        if any(record["returncode"] != 0 for record in result["build"]):
            raise RuntimeError("One or more ladder builds failed")
        if any(record["returncode"] != 0 for record in result["commands"]):
            raise RuntimeError("One or more ladder runs failed; see run.json")
    except Exception as exc:
        (out / "error.txt").write_text(str(exc) + "\n")
        raise
