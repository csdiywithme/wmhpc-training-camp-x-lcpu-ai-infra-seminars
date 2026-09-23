"""Observe unmodified 3.3 with short per-process timeouts on one B300.
Use MODAL_PROFILE=simidawhu. Only INPUTS and this runner are uploaded.
"""
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
CUDA = HERE.parent / "cuda"
ROOT = Path("/opt/m3_mbarrier/cuda")
INPUTS = ("common.h", "Makefile", "m3_tcgen05/03_bug_mbarrier.cu", "m3_tcgen05/judge_mbar.sh")


def command(argv, timeout=60):
    try:
        p = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True,
                           timeout=timeout)
        result = dict(command=argv, returncode=p.returncode,
                      stdout=p.stdout, stderr=p.stderr)
    except subprocess.TimeoutExpired:
        result = dict(command=argv, returncode=None, error="timeout")
    print(json.dumps(result), flush=True)
    return result


def build():
    records = []
    for argv in (["nvcc", "--version"],
                 ["make", "-B", "ARCH=100f", "bin/m3_tcgen05/03_bug_mbarrier", "ptx/m3_tcgen05/03_bug_mbarrier"]):
        record = command(argv, 120)
        records.append(record)
        if record["returncode"] != 0:
            raise RuntimeError("Compilation failed; see command output")
    (ROOT / "build.json").write_text(json.dumps(records, indent=2))


image = (modal.Image.from_registry("nvidia/cuda:13.1.0-devel-ubuntu24.04",
                                  add_python="3.11")
         .entrypoint([]).apt_install("build-essential", "cuda-nsight-compute-13-1"))
for name in INPUTS:
    image = image.add_local_file(CUDA / name, str(ROOT / name), copy=True)
image = image.run_function(build, timeout=360)
app = modal.App("assignment02-m3-mbarrier")


@app.function(image=image, gpu="B300", timeout=90, retries=0,
              max_containers=1, scaledown_window=2)
def run():
    records = []
    records.append(command(["nvidia-smi"], 10))
    for rounds in (1, 2, 4):
        for seed in (42, 7):
            record = command(["timeout", "-k", "2s", "5s",
                              "./bin/m3_tcgen05/03_bug_mbarrier", str(seed), str(rounds)], 10)
            record.update(rounds=rounds, seed=seed,
                          outcome="PASS" if record["returncode"] == 0 else
                                  "TIMEOUT" if record["returncode"] in (124, 137) else "ERROR")
            records.append(record)
    return dict(commands=records,
                ptx=(ROOT / "m3_tcgen05/03_bug_mbarrier.ptx").read_text(),
                build=json.loads((ROOT / "build.json").read_text()),
                input_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                              for name in INPUTS})


@app.local_entrypoint()
def main():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = HERE / "results" / ("m3-mbarrier-" + stamp)
    out.mkdir(parents=True)
    for name in INPUTS:
        target = out / "inputs" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((CUDA / name).read_bytes())
    (out / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    try:
        result = run.remote()
        (out / "run.json").write_text(json.dumps(result, indent=2) + "\n")
        (out / "03_bug_mbarrier.ptx").write_text(result["ptx"])
        for name in INPUTS:
            if hashlib.sha256((out / "inputs" / name).read_bytes()).hexdigest() != result["input_sha256"][name]:
                raise RuntimeError("Input snapshot differs from remote source")
        print("Saved:", out)
        print("Baseline outcomes:", [(r.get("rounds"), r.get("seed"), r.get("outcome"))
                                     for r in result["commands"] if "rounds" in r])
    except Exception as exc:
        (out / "error.txt").write_text(str(exc) + "\n")
        raise
