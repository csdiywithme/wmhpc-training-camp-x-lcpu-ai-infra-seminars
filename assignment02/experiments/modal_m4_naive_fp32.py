"""Bounded B300 magnitude comparison for Assignment01's original naive FP32 kernel."""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
ROOT = Path("/opt/m4_naive_fp32")
INPUTS = (
    "assignment01/cuda/common.h",
    "assignment01/cuda/bonus/matmul.cu",
    "assignment02/experiments/m4_naive_fp32_4096.cu",
)


def command(argv, timeout=90):
    try:
        proc = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True,
                              timeout=timeout)
        record = dict(command=argv, returncode=proc.returncode,
                      stdout=proc.stdout, stderr=proc.stderr)
    except subprocess.TimeoutExpired:
        record = dict(command=argv, returncode=None, error="timeout")
    print(json.dumps(record), flush=True)
    return record


def build():
    record = command([
        "nvcc", "-O2", "-std=c++17", "-I", str(ROOT / "assignment01/cuda"),
        "-gencode", "arch=compute_100f,code=sm_100f", "-o", str(ROOT / "naive4096"),
        "assignment02/experiments/m4_naive_fp32_4096.cu",
    ], 120)
    if record["returncode"] != 0:
        raise RuntimeError("A01 naive wrapper compilation failed")
    (ROOT / "build.json").write_text(json.dumps(record, indent=2))


image = (modal.Image.from_registry("nvidia/cuda:13.1.0-devel-ubuntu24.04",
                                   add_python="3.11")
         .entrypoint([]).apt_install("build-essential"))
for name in INPUTS:
    image = image.add_local_file(REPO / name, str(ROOT / name), copy=True)
image = image.run_function(build, timeout=180)
app = modal.App("assignment02-m4-naive-fp32")


@app.function(image=image, gpu="B300", timeout=30, retries=0,
              max_containers=1, scaledown_window=2)
def run():
    records = [command(["nvidia-smi"], 10)]
    records.append(command(["timeout", "-k", "2s", "5s", "./naive4096"], 10))
    records.append(command(["nvidia-smi", "--query-compute-apps=pid,process_name",
                            "--format=csv"], 10))
    return dict(commands=records,
                build=json.loads((ROOT / "build.json").read_text()),
                input_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                              for name in INPUTS})


@app.local_entrypoint()
def main():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = HERE / "results" / ("m4-naive-fp32-" + stamp)
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
        if result["build"]["returncode"] != 0 or result["commands"][1]["returncode"] != 0:
            raise RuntimeError("Naive FP32 comparison failed; see run.json")
    except Exception as exc:
        (out / "error.txt").write_text(str(exc) + "\n")
        raise
