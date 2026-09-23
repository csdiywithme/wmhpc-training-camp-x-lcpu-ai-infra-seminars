"""Profile the user-written 4.3 S=3 pipeline GEMM and download the native NCU report.
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
ROOT = Path("/opt/m4_pipeline_ncu/cuda")
INPUTS = ("common.h", "Makefile", "m4_gemm/03_pipeline.cu", "m4_gemm/02_tma.cu")


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
                 ["make", "-B", "ARCH=100f", "FLAGS=-O2 -std=c++17 -I. --expt-relaxed-constexpr -lineinfo -DSTAGES=3", "bin/m4_gemm/03_pipeline", "ptx/m4_gemm/03_pipeline", "bin/m4_gemm/02_tma"]):
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
app = modal.App("assignment02-m4-pipeline-ncu")


@app.function(image=image, gpu="B300", timeout=120, retries=0,
              max_containers=1, scaledown_window=2)
def run():
    records = []
    records.append(command(["nvidia-smi"], 10))
    records.append(command(["ncu", "--version"], 10))
    records.append(command(["timeout", "-k", "2s", "5s", "./bin/m4_gemm/02_tma", "4096", "4096", "4096"], 10))
    for shape in ((128, 64, 64), (128, 64, 128), (256, 192, 256), (4096, 4096, 4096)):
        records.append(command(["timeout", "-k", "2s", "5s", "./bin/m4_gemm/03_pipeline", *map(str, shape)], 10))
        if records[-1]["returncode"] != 0:
            break
    report = ROOT / "pipeline_4096.ncu-rep"
    if all(r["returncode"] == 0 for r in records):
        records.append(command(["timeout", "-k", "2s", "35s", "ncu",
                                "--clock-control", "none", "--set", "full",
                                "--kernel-name", "regex:gemm_pipeline", "--launch-skip", "21",
                                "--launch-count", "1", "--import-source", "yes",
                                "--source-folders", str(ROOT), "--export", str(report),
                                "./bin/m4_gemm/03_pipeline", "4096", "4096", "4096"], 40))
    exports = {}
    if report.exists():
        for page in ("details", "raw", "source"):
            argv = ["ncu", "--import", str(report), "--page", page]
            if page == "raw": argv += ["--csv"]
            p = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=15)
            exports[page] = dict(command=argv, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
    records.append(command(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv"], 10))
    return dict(commands=records, exports=exports,
                report_bytes=report.read_bytes() if report.exists() else b"",
                ptx=(ROOT / "m4_gemm/03_pipeline.ptx").read_text(),
                build=json.loads((ROOT / "build.json").read_text()),
                input_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                              for name in INPUTS})


@app.local_entrypoint()
def main():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = HERE / "results" / ("m4-pipeline-ncu-" + stamp)
    out.mkdir(parents=True)
    for name in INPUTS:
        target = out / "inputs" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((CUDA / name).read_bytes())
    (out / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    try:
        result = run.remote()
        report_bytes = result.pop("report_bytes")
        if report_bytes:
            (out / "pipeline_4096.ncu-rep").write_bytes(report_bytes)
            result["report_sha256"] = hashlib.sha256(report_bytes).hexdigest()
            result["report_size"] = len(report_bytes)
        for page, data in result["exports"].items():
            (out / ("ncu-" + page + (".csv" if page == "raw" else ".txt"))).write_text(data["stdout"])
        (out / "run.json").write_text(json.dumps(result, indent=2) + "\n")
        (out / "03_pipeline.ptx").write_text(result["ptx"])
        for name in INPUTS:
            if hashlib.sha256((out / "inputs" / name).read_bytes()).hexdigest() != result["input_sha256"][name]:
                raise RuntimeError("Input snapshot differs from remote source")
        print("Saved:", out)
        if any(r["returncode"] != 0 for r in result["commands"]):
            raise RuntimeError("Command failed; see run.json")
    except Exception as exc:
        (out / "error.txt").write_text(str(exc) + "\n")
        raise
