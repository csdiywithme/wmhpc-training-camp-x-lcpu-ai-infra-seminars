"""Bounded C2 calibration and post-profile candidate experiments on B300."""
from datetime import datetime, timezone
import json
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parent
image = (modal.Image.from_registry("nvidia/cuda:13.1.0-devel-ubuntu24.04", add_python="3.11")
         .entrypoint([]).apt_install("git", "cuda-nsight-compute-13-1")
         .pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cu130")
         .pip_install("numpy==2.2.6"))
FILES = ("candidate.py", "harness/synth.py", "harness/vllm_shim.py", "vllm_msa_ref/sparse_attn.py",
         "experiments/baseline_workload.py", "experiments/candidate_workload.py",
         "validation/__init__.py", "validation/suite.py", "validation/run_validation.py",
         "validation/ACCEPTANCE.md")
for name in FILES:
    image = image.add_local_file(ROOT / name, f"/opt/c2/{name}")
app = modal.App("c2-candidate-validation")


@app.function(image=image, gpu="B300", cpu=4, memory=16384, timeout=1200,
              retries=0, max_containers=1, scaledown_window=2)
def experiment(mode="calibrate", manifest=None):
    import subprocess
    import time
    out = Path("/tmp/c2-candidate-artifacts")
    out.mkdir(exist_ok=True)
    records = []
    def run(label, argv, limit):
        start = time.monotonic()
        try:
            p = subprocess.run(argv, cwd="/opt/c2", capture_output=True, text=True, timeout=limit)
            r = dict(label=label, command=argv, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
        except subprocess.TimeoutExpired as e:
            r = dict(label=label, command=argv, returncode=None, error="timeout",
                     stdout=e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else e.stdout,
                     stderr=e.stderr.decode(errors="replace") if isinstance(e.stderr, bytes) else e.stderr)
        r["seconds"] = time.monotonic() - start
        records.append(r)
        (out / "commands.json").write_text(json.dumps(records, indent=2) + "\n")
        print(f"{label}: rc={r['returncode']}, seconds={r['seconds']:.1f}", flush=True)
        if r["returncode"] != 0:
            print(str(r.get("stdout", ""))[-5000:], str(r.get("stderr", ""))[-5000:], flush=True)
        return r["returncode"] == 0
    run("environment", ["nvidia-smi"], 15)
    validation = ["python", "-u", "validation/run_validation.py"]
    if mode == "calibrate":
        run("calibrate-full", validation + ["calibrate", "--tier", "full", "--output", str(out / "calibration.json")], 1000)
    elif mode == "verify":
        if not manifest:
            raise ValueError("verify requires the saved frozen baseline manifest")
        (out / "calibration.json").write_text(manifest)
        run("verify-full-candidate", validation + ["verify", "--manifest", str(out / "calibration.json"),
            "--adapter", "candidate:run", "--output", str(out / "candidate-heldout.json")], 1050)
    elif mode in ("tune", "smoke", "merge", "paired"):
        cmd = ["python", "-u", "experiments/candidate_workload.py", "--output", str(out)]
        if mode == "smoke":
            cmd += ["--batch", "1", "--tp", "4", "--dtype", "bf16"]
        if mode == "merge":
            cmd += ["--mode", "merge"]
        if mode == "paired":
            cmd += ["--mode", "paired"]
        run(mode, cmd, 1050 if mode in ("tune", "merge", "paired") else 400)
    elif mode == "profile":
        import shutil
        ncu=shutil.which("ncu")
        if ncu is None:
            raise RuntimeError("NCU unavailable")
        run("selected-profile", [ncu,"--set","detailed","--kernel-name-base","function",
            "--kernel-name","regex:(_page_decode_kernel|_merge_feature_tiles)",
            "--launch-count","4","--clock-control","none","--cache-control","all",
            "--force-overwrite","--export",str(out/"selected"),
            "python","-u","experiments/candidate_workload.py","--mode","profile","--output",str(out)],600)
        if (out/"selected.ncu-rep").exists():
            run("selected-csv",[ncu,"--import",str(out/"selected.ncu-rep"),"--page","raw","--csv"],60)
            (out/"selected.csv").write_text(records[-1].get("stdout", ""))
    else:
        raise ValueError(mode)
    return {str(p.relative_to(out)): p.read_bytes() for p in out.rglob("*") if p.is_file()}


@app.local_entrypoint()
def main(mode: str = "calibrate", manifest: str = ""):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = ROOT / "results" / f"candidate-{mode}-{stamp}"
    folder.mkdir(parents=True, exist_ok=False)
    for name in FILES:
        target = folder / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    print(f"C2_CANDIDATE_OUTPUT={folder}", flush=True)
    payload = Path(manifest).read_text() if manifest else None
    result = experiment.remote(mode, payload)
    for name, content in result.items():
        target = folder / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    print(f"Saved {folder}")
    records = json.loads((folder / "commands.json").read_text())
    if any(row["returncode"] != 0 for row in records):
        raise RuntimeError("Experiment failed; persisted original output must be inspected")
