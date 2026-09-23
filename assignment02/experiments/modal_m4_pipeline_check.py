"""Short, scoped B300 correctness check for the in-progress 4.3 kernel."""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
CUDA = HERE.parent / "cuda"
ROOT = Path("/opt/m4_pipeline/cuda")
INPUTS = ("common.h", "Makefile", "m4_gemm/03_pipeline.cu")


def command(argv, timeout=60):
    try:
        proc = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True,
                              timeout=timeout)
        result = dict(command=argv, returncode=proc.returncode,
                      stdout=proc.stdout, stderr=proc.stderr)
    except subprocess.TimeoutExpired:
        result = dict(command=argv, returncode=None, error="timeout")
    print(json.dumps(result), flush=True)
    return result


def build():
    result = command(["make", "-B", "ARCH=100f", "STAGES=3",
                      "bin/m4_gemm/03_pipeline"], 120)
    if result["returncode"] != 0:
        raise RuntimeError("Compilation failed")


image = (modal.Image.from_registry("nvidia/cuda:13.1.0-devel-ubuntu24.04",
                                   add_python="3.11")
         .entrypoint([]).apt_install("build-essential"))
for name in INPUTS:
    image = image.add_local_file(CUDA / name, str(ROOT / name), copy=True)
image = image.run_function(build, timeout=180)
app = modal.App("assignment02-m4-pipeline-check")


@app.function(image=image, gpu="B300", timeout=90, retries=0,
              max_containers=1, scaledown_window=2)
def run():
    records = [command(["nvidia-smi"], 10)]
    for shape in ((128, 64, 64), (128, 64, 128), (256, 192, 256),
                  (4096, 4096, 4096)):
        records.append(command(["timeout", "-k", "2s", "5s",
                                "./bin/m4_gemm/03_pipeline", *map(str, shape)], 10))
        if records[-1]["returncode"] != 0:
            break
    return dict(commands=records,
                input_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                              for name in INPUTS})


@app.local_entrypoint()
def main():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = HERE / "results" / ("m4-pipeline-check-" + stamp)
    out.mkdir(parents=True)
    for name in INPUTS:
        target = out / "inputs" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((CUDA / name).read_bytes())
    (out / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    try:
        result = run.remote()
        (out / "run.json").write_text(json.dumps(result, indent=2) + "\n")
        for name in INPUTS:
            digest = hashlib.sha256((out / "inputs" / name).read_bytes()).hexdigest()
            if digest != result["input_sha256"][name]:
                raise RuntimeError("Input snapshot differs from remote source")
        print("Saved:", out)
        if any(r["returncode"] != 0 for r in result["commands"]):
            raise RuntimeError("Correctness check failed; see run.json")
    except Exception as exc:
        (out / "error.txt").write_text(str(exc) + "\n")
        raise
