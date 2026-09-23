"""One bounded B300 experiment for C2 paged copy/TMA feasibility."""
import json
from datetime import datetime, timezone
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
image = (
    modal.Image.from_registry("nvidia/cuda:13.1.0-devel-ubuntu24.04", add_python="3.11")
    .entrypoint([])
    .add_local_file(HERE / "experiments/paged_copy.cu", "/opt/paged_copy.cu", copy=True)
    .run_commands("nvcc -O3 -lineinfo -arch=sm_103a /opt/paged_copy.cu -L/usr/local/cuda/lib64/stubs -lcuda -o /opt/paged_copy")
)
app = modal.App("c2-paged-copy")


@app.function(image=image, gpu="B300", cpu=2, memory=4096, timeout=240,
              retries=0, max_containers=1, scaledown_window=2)
def experiment():
    import hashlib
    import subprocess
    records = []
    for argv in (["nvidia-smi"], ["nvcc", "--version"], ["/opt/paged_copy"],
                 ["cuobjdump", "-sass", "/opt/paged_copy"]):
        try:
            p = subprocess.run(argv, capture_output=True, text=True, timeout=170)
            records.append(dict(command=argv, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr))
        except subprocess.TimeoutExpired as e:
            records.append(dict(command=argv, returncode=None, error="timeout",
                                stdout=(e.stdout or b"").decode() if isinstance(e.stdout, bytes) else e.stdout))
    return {"source_sha256": hashlib.sha256(Path("/opt/paged_copy.cu").read_bytes()).hexdigest(), "records": records}


@app.local_entrypoint()
def main():
    result = experiment.remote()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = HERE / "results" / f"paged-copy-{stamp}"
    folder.mkdir(parents=True, exist_ok=True)
    for record in result["records"]:
        if record["command"][0] == "cuobjdump":
            (folder / "paged_copy.sass").write_text(record.pop("stdout", ""))
            record["stdout_file"] = "paged_copy.sass"
    (folder / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {folder}")
    for record in result["records"]:
        if record["command"] == ["/opt/paged_copy"]:
            print(record.get("stdout", ""))
    if any(record["returncode"] != 0 for record in result["records"]):
        raise RuntimeError("Experiment failed; inspect persisted raw output")
