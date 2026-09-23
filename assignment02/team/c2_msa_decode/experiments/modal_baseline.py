"""Short single-GPU C2 jobs with raw, locally persisted measurement artifacts."""
from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GPU = os.environ.get("C2_GPU", "B300")
BASE_IMAGE = "nvidia/cuda:13.1.0-devel-ubuntu24.04"
image = (modal.Image.from_registry(BASE_IMAGE, add_python="3.11").entrypoint([])
         .apt_install("git", "cuda-nsight-compute-13-1")
         .pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cu130")
         .pip_install("numpy==2.2.6"))
INPUT_FILES = ("harness/synth.py", "harness/vllm_shim.py",
               "vllm_msa_ref/sparse_attn.py", "experiments/baseline_workload.py")
for filename in INPUT_FILES:
    image = image.add_local_file(ROOT / filename, f"/opt/c2/{filename}")
app = modal.App("c2-msa-baseline")


@app.function(image=image, gpu=GPU, cpu=4, memory=16384, timeout=600,
              retries=0, max_containers=1, scaledown_window=2)
def remote_run(mode="bench", batch=1, tp=4, dtype="bf16", pdl=False):
    import glob
    import shutil
    output = Path("/tmp/c2-artifacts")
    output.mkdir(exist_ok=True)
    records = []
    def run(label, argv, timeout=30):
        begin = time.monotonic()
        try:
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
            row = dict(label=label, command=argv, returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
        except subprocess.TimeoutExpired as exc:
            row = dict(label=label, command=argv, returncode=None, error="timeout",
                       stdout=exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout,
                       stderr=exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr)
        row["elapsed_seconds"] = time.monotonic() - begin
        records.append(row)
        (output / "commands.json").write_text(json.dumps(records, indent=2) + "\n")
        print(json.dumps(row), flush=True)
        return row
    run("nvidia-smi-before", ["nvidia-smi"])
    run("nvcc", ["nvcc", "--version"])
    run("pip-freeze", ["python", "-m", "pip", "freeze"])
    settings = Path("/proc/driver/nvidia/params")
    if settings.exists():
        (output / "profiling-driver-settings.txt").write_text("\n".join(line for line in settings.read_text().splitlines() if "Profil" in line))
    script = ["python", "-u", "/opt/c2/experiments/baseline_workload.py", "--output", str(output)]
    shape = ["--batch", str(batch), "--tp", str(tp), "--dtype", dtype]
    if pdl:
        shape += ["--pdl"]
    if mode == "bench":
        run("baseline-matrix", script + ["--all"] + (["--pdl"] if pdl else []), 480)
    elif mode == "one":
        run("baseline-one", script + shape, 180)
    elif mode in ("profile", "profile-matrix"):
        candidates = sorted(glob.glob("/opt/nvidia/nsight-compute/*/ncu"))
        ncu = shutil.which("ncu") or (candidates[-1] if candidates else None)
        if not ncu:
            raise RuntimeError("ncu is not installed")
        run("ncu-version", [ncu, "--version"])
        cases = [(tp, batch, dtype)] if mode == "profile" else [(t, b, d) for t, b in [(1, 1), (1, 16), (4, 16)] for d in ["bf16", "fp8"]]
        for case_tp, case_batch, case_dtype in cases:
            key = f"tp{case_tp}-b{case_batch}-{case_dtype}"
            case_out = output if mode == "profile" else output / key
            case_out.mkdir(exist_ok=True)
            case_shape = ["--batch", str(case_batch), "--tp", str(case_tp), "--dtype", case_dtype] + (["--pdl"] if pdl else [])
            case_script = ["python", "-u", "/opt/c2/experiments/baseline_workload.py", "--output", str(case_out)]
            run(f"ncu-baseline-{key}", [ncu, "--set", "detailed", "--kernel-name-base", "function",
                            "--kernel-name", "regex:(_gqa_sparse_decode_kernel|_merge_topk_attn_out_kernel)",
                            "--launch-count", "2", "--clock-control", "none", "--cache-control", "all",
                            "--force-overwrite", "--export", str(case_out / "baseline"),
                            *case_script, "--mode", "profile", *case_shape], 70)
            report = case_out / "baseline.ncu-rep"
            if report.exists():
                row = run(f"ncu-csv-{key}", [ncu, "--import", str(report), "--page", "raw", "--csv"], 30)
                (case_out / "baseline.csv").write_text(row["stdout"] or "")
    else:
        raise ValueError(mode)
    run("nvidia-smi-after", ["nvidia-smi"])
    return dict(commands=records, gpu_requested=GPU, base_image=BASE_IMAGE,
                artifacts={str(p.relative_to(output)): p.read_bytes() for p in output.rglob("*") if p.is_file()})


@app.local_entrypoint()
def main(mode: str = "bench", batch: int = 1, tp: int = 4, dtype: str = "bf16", pdl: bool = False):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = HERE / "results" / f"{mode}-{GPU.lower()}-{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "modal_baseline.py").write_bytes(Path(__file__).read_bytes())
    (output / "baseline_workload.py").write_bytes((HERE / "baseline_workload.py").read_bytes())
    print(f"C2_OUTPUT={output}", flush=True)
    try:
        result = remote_run.remote(mode, batch, tp, dtype, pdl)
    except Exception as exc:
        (output / "error.json").write_text(json.dumps(dict(error_type=type(exc).__name__, error=str(exc)), indent=2) + "\n")
        raise
    for name, content in result.pop("artifacts").items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    (output / "run.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"C2_SAVED={output}", flush=True)
    if any(row["returncode"] != 0 for row in result["commands"]):
        raise RuntimeError("One or more subprocesses failed; raw output saved.")
