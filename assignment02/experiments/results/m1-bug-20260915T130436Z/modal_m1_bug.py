"""Run exercise 1.2 unchanged, then print matrices from a diagnostic copy.

Select the authenticated simidawhu profile explicitly with MODAL_PROFILE.
Only common.h, Makefile, this exercise, and this runner are uploaded.
The diagnostic copy adds host printing only; no kernel or input is changed.
"""
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
CUDA = HERE.parent / "cuda"
ROOT = Path("/opt/m1/cuda")
INPUTS = ("common.h", "Makefile", "m1_sm80/02_bug_fragment.cu")


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
                 ["make", "-B", "ARCH=100f", "bin/m1_sm80/02_bug_fragment"]):
        record = command(argv, 120)
        records.append(record)
        if record["returncode"] != 0:
            raise RuntimeError("Compilation failed; see command output")
    source = (ROOT / INPUTS[2]).read_text()
    marker = "    long bad = 0;"
    if source.count(marker) != 1:
        raise RuntimeError("Diagnostic insertion point is ambiguous")
    printing = '''    for (int r = 0; r < 16; ++r) {
        printf("GOT row %d:", r);
        for (int n = 0; n < 8; ++n) printf(" %.0f", got[r * 8 + n]);
        printf("\\nREF row %d:", r);
        for (int n = 0; n < 8; ++n) printf(" %.0f", ref[r * 8 + n]);
        printf("\\n");
    }
'''
    diagnostic = ROOT / "m1_sm80/02_bug_fragment_diagnostic.cu"
    diagnostic.write_text(source.replace(marker, printing + marker))
    record = command(["make", "-B", "ARCH=100f",
                      "bin/m1_sm80/02_bug_fragment_diagnostic"], 120)
    records.append(record)
    if record["returncode"] != 0:
        raise RuntimeError("Diagnostic compilation failed")
    (ROOT / "build.json").write_text(json.dumps(records, indent=2))


image = (modal.Image.from_registry("nvidia/cuda:13.1.0-devel-ubuntu24.04",
                                  add_python="3.11")
         .entrypoint([]).apt_install("build-essential"))
for name in INPUTS:
    image = image.add_local_file(CUDA / name, str(ROOT / name), copy=True)
image = image.run_function(build, timeout=360)
app = modal.App("assignment02-m1-bug")


@app.function(image=image, gpu="B300", timeout=120, retries=0,
              max_containers=1, scaledown_window=2)
def run():
    records = []
    for argv in (["nvidia-smi"],
                 [str(ROOT / "bin/m1_sm80/02_bug_fragment")],
                 [str(ROOT / "bin/m1_sm80/02_bug_fragment_diagnostic")]):
        records.append(command(argv, 20))
    return dict(commands=records,
                build=json.loads((ROOT / "build.json").read_text()),
                input_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                              for name in INPUTS},
                diagnostic_source=(ROOT / "m1_sm80/02_bug_fragment_diagnostic.cu").read_text())


@app.local_entrypoint()
def main():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = HERE / "results" / ("m1-bug-" + stamp)
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
            if hashlib.sha256((out / "inputs" / name).read_bytes()).hexdigest() != result["input_sha256"][name]:
                raise RuntimeError("Input snapshot differs from remote source")
        (out / "02_bug_fragment_diagnostic.cu").write_text(result["diagnostic_source"])
        print("Saved:", out)
    except Exception as exc:
        (out / "error.txt").write_text(str(exc) + "\n")
        raise
