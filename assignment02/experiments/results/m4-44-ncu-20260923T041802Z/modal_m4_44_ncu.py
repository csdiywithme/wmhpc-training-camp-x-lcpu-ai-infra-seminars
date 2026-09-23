"""Capture local Nsight Compute reports for the retained and 2-CTA 4.4 variants."""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
CUDA = HERE.parent / "cuda"
ROOT = Path("/opt/m4_44_ncu/cuda")
STEMS = ("04ab_warp_persistent", "04c_cta_pair")
INPUTS = ("common.h", "Makefile") + tuple(f"m4_gemm/{stem}.cu" for stem in STEMS)


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
    for stem in STEMS:
        record = command(["make", "-B", "ARCH=100f",
                          "FLAGS=-O2 -std=c++17 -I. --expt-relaxed-constexpr -lineinfo -DSTAGES=3",
                          f"bin/m4_gemm/{stem}"], 120)
        records.append(record)
        if record["returncode"] != 0:
            raise RuntimeError(f"{stem} build failed")
    (ROOT / "build.json").write_text(json.dumps(records, indent=2))


image = (modal.Image.from_registry("nvidia/cuda:13.1.0-devel-ubuntu24.04",
                                   add_python="3.11")
         .entrypoint([]).apt_install("build-essential", "cuda-nsight-compute-13-1"))
for name in INPUTS:
    image = image.add_local_file(CUDA / name, str(ROOT / name), copy=True)
image = image.run_function(build, timeout=300)
app = modal.App("assignment02-m4-44-ncu")


@app.function(image=image, gpu="B300", timeout=120, retries=0,
              max_containers=1, scaledown_window=2)
def run():
    records = [command(["nvidia-smi"], 10), command(["ncu", "--version"], 10)]
    reports = {}
    exports = {}
    for stem in STEMS:
        report = ROOT / f"{stem}_4096.ncu-rep"
        records.append(command(["timeout", "-k", "2s", "35s", "ncu",
                                "--clock-control", "none", "--set", "full",
                                "--kernel-name", "regex:gemm_pipeline",
                                "--launch-skip", "21", "--launch-count", "1",
                                "--import-source", "yes", "--source-folders", str(ROOT),
                                "--export", str(report),
                                f"./bin/m4_gemm/{stem}", "4096", "4096", "4096"], 40))
        if report.exists():
            reports[stem] = report.read_bytes()
            exports[stem] = {}
            for page in ("details", "raw", "source"):
                argv = ["ncu", "--import", str(report), "--page", page]
                if page == "raw":
                    argv += ["--csv"]
                exports[stem][page] = command(argv, 15)
    return dict(commands=records, reports=reports, exports=exports,
                build=json.loads((ROOT / "build.json").read_text()),
                input_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                              for name in INPUTS})


@app.local_entrypoint()
def main():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = HERE / "results" / ("m4-44-ncu-" + stamp)
    out.mkdir(parents=True)
    for name in INPUTS:
        target = out / "inputs" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((CUDA / name).read_bytes())
    (out / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    try:
        result = run.remote()
        reports = result.pop("reports")
        result["report_sha256"] = {}
        result["report_size"] = {}
        for stem, data in reports.items():
            (out / f"{stem}_4096.ncu-rep").write_bytes(data)
            result["report_sha256"][stem] = hashlib.sha256(data).hexdigest()
            result["report_size"][stem] = len(data)
        for stem, pages in result["exports"].items():
            for page, record in pages.items():
                (out / f"{stem}-ncu-{page}{'.csv' if page == 'raw' else '.txt'}").write_text(record["stdout"])
        (out / "run.json").write_text(json.dumps(result, indent=2) + "\n")
        for name in INPUTS:
            digest = hashlib.sha256((out / "inputs" / name).read_bytes()).hexdigest()
            if digest != result["input_sha256"][name]:
                raise RuntimeError(f"Input snapshot differs from remote source: {name}")
        if any(r["returncode"] != 0 for r in result["commands"]):
            raise RuntimeError("One or more NCU commands failed; see run.json")
        print("Saved:", out)
    except Exception as exc:
        (out / "error.txt").write_text(str(exc) + "\n")
        raise
