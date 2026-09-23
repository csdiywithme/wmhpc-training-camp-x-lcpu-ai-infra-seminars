"""Scoped same-allocation B300 compare of 4.3 and one saved 4.4 variant."""

import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
CUDA = HERE.parent / "cuda"
ROOT = Path("/opt/m4_44/cuda")
VARIANT = os.environ.get("M4_VARIANT", "04a_warp_specialized")
if not re.fullmatch(r"04[a-z0-9_]+", VARIANT):
    raise ValueError("M4_VARIANT must name one m4_gemm/04*.cu variant")
REFERENCE = os.environ.get("M4_REFERENCE", "03_pipeline")
if REFERENCE != "03_pipeline" and not re.fullmatch(r"04[a-z0-9_]+", REFERENCE):
    raise ValueError("M4_REFERENCE must name 03_pipeline or one 04*.cu variant")
INPUTS = tuple(dict.fromkeys(("common.h", "Makefile",
                              f"m4_gemm/{REFERENCE}.cu",
                              f"m4_gemm/{VARIANT}.cu")))


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
    records = []
    stems = sorted(path.stem for path in (ROOT / "m4_gemm").glob("*.cu"))
    if len(stems) < 2:
        raise RuntimeError(f"Expected a reference and candidate source, found {stems}")
    for stem in stems:
        record = command(["make", "-B", "ARCH=100f", "STAGES=3",
                          f"bin/m4_gemm/{stem}"], 120)
        records.append(record)
        if record["returncode"] != 0:
            raise RuntimeError(f"{stem} compilation failed")
    (ROOT / "build.json").write_text(json.dumps(records, indent=2))


image = (modal.Image.from_registry("nvidia/cuda:13.1.0-devel-ubuntu24.04",
                                   add_python="3.11")
         .entrypoint([]).apt_install("build-essential"))
for name in INPUTS:
    image = image.add_local_file(CUDA / name, str(ROOT / name), copy=True)
image = image.run_function(build, timeout=300)
app = modal.App("assignment02-m4-44-compare")


@app.function(image=image, gpu="B300", timeout=120, retries=0,
              max_containers=1, scaledown_window=2)
def run(reference, candidate):
    for stem in (reference, candidate):
        if not (ROOT / f"m4_gemm/{stem}.cu").exists():
            raise RuntimeError(f"Missing source for {stem}")
    records = [command(["nvidia-smi"], 10)]
    if "two_consumer" in candidate:
        smoke_shapes = ((512, 64, 64), (512, 64, 128),
                        (512, 192, 256), (512, 4096, 16384))
    elif "cta_pair" in candidate:
        smoke_shapes = ((256, 64, 64), (256, 64, 128),
                        (256, 192, 256), (256, 4096, 16384))
    else:
        smoke_shapes = ((128, 64, 64), (128, 64, 128),
                        (256, 192, 256), (256, 4096, 16384))
    for shape in smoke_shapes:
        record = command(["timeout", "-k", "2s", "5s",
                          f"./bin/m4_gemm/{candidate}", *map(str, shape)], 10)
        records.append(record)
        if record["returncode"] != 0:
            break
    else:
        for _ in range(3):
            for stem in (reference, candidate):
                records.append(command(["timeout", "-k", "2s", "5s",
                                        f"./bin/m4_gemm/{stem}",
                                        "4096", "4096", "4096"], 10))
                if records[-1]["returncode"] != 0:
                    break
            if records[-1]["returncode"] != 0:
                break
    records.append(command(["nvidia-smi", "--query-compute-apps=pid,process_name",
                            "--format=csv"], 10))
    remote_inputs = ("common.h", "Makefile") + tuple(
        str(path.relative_to(ROOT)) for path in sorted((ROOT / "m4_gemm").glob("*.cu")))
    return dict(reference=reference, variant=candidate, commands=records,
                build=json.loads((ROOT / "build.json").read_text()),
                input_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                              for name in remote_inputs})


@app.local_entrypoint()
def main():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = HERE / "results" / ("m4-44-" + VARIANT + "-" + stamp)
    out.mkdir(parents=True)
    for name in INPUTS:
        target = out / "inputs" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((CUDA / name).read_bytes())
    (out / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    try:
        result = run.remote(REFERENCE, VARIANT)
        (out / "run.json").write_text(json.dumps(result, indent=2) + "\n")
        for name in INPUTS:
            digest = hashlib.sha256((out / "inputs" / name).read_bytes()).hexdigest()
            if digest != result["input_sha256"][name]:
                raise RuntimeError("Input snapshot differs from remote source")
        print("Saved:", out)
        if any(r["returncode"] != 0 for r in result["commands"]):
            raise RuntimeError("One or more candidate runs failed; see run.json")
    except Exception as exc:
        (out / "error.txt").write_text(str(exc) + "\n")
        raise
