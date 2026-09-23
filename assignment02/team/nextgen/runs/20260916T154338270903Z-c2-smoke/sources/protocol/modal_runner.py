"""Immutable source bundle -> CPU compilation -> bounded GPU experiment.

Set NEXTGEN_RUN_DIR to a directory created by prepare_run.py and explicitly use
MODAL_PROFILE=simidawhu. No compilation failure is allowed to start a GPU job.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import os
import sys
from pathlib import Path
import tarfile

import modal

if sys.version_info[:2] != (3, 11):
    raise RuntimeError("Use local Python 3.11 to match serialized remote functions")

HERE = Path(__file__).resolve().parent
RUN_DIR = Path(os.environ["NEXTGEN_RUN_DIR"]).resolve()
REQUEST = json.loads((RUN_DIR / "request.json").read_text())
if os.environ.get("MODAL_PROFILE") != "simidawhu":
    raise RuntimeError("Explicit MODAL_PROFILE=simidawhu is required")
if REQUEST["gpu"] != "B300":
    raise RuntimeError("B200 baseline image must be rebuilt for sm_100a before using this runner")

spec = importlib.util.spec_from_file_location("nextgen_base", RUN_DIR / "sources/protocol/base_image.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
image = base.base_image.env({"PYTHONPATH": "/tmp/nextgen-build:/opt/c2:/opt/nextgen/c1:/opt/c1_reference",
                             "TORCH_CUDA_ARCH_LIST": REQUEST["arch"].replace("103a", "10.3a"),
                             "MAX_JOBS": "4", "NVCC_THREADS": "2"})
for directory, remote in (("c1", "/opt/nextgen/c1"), ("c1_reference", "/opt/c1_reference"),
                          ("c2", "/opt/c2"), ("protocol", "/opt/nextgen/protocol")):
    if (RUN_DIR / "sources" / directory).exists():
        image = image.add_local_dir(RUN_DIR / "sources" / directory, remote)
image = image.add_local_file(RUN_DIR / "source_manifest.json", "/opt/nextgen/source_manifest.json")
app = modal.App("nextgen-c1-c2-exploration")


def command(argv, *, timeout, cwd=None):
    import signal
    import subprocess
    import time
    start = time.monotonic()
    try:
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, cwd=cwd, start_new_session=True)
        stdout, stderr = proc.communicate(timeout=timeout)
        record = {"command": argv, "returncode": proc.returncode,
                  "stdout": stdout, "stderr": stderr}
    except subprocess.TimeoutExpired as exc:
        os.killpg(proc.pid, signal.SIGKILL)
        stdout, stderr = proc.communicate()
        record = {"command": argv, "returncode": None, "error": "timeout",
                  "stdout": stdout, "stderr": stderr}
    record["seconds"] = time.monotonic() - start
    print(json.dumps({"command": argv, "returncode": record["returncode"],
                      "seconds": record["seconds"]}), flush=True)
    if record["returncode"] != 0:
        print((record.get("stdout") or "")[-4000:], (record.get("stderr") or "")[-6000:], flush=True)
    return record


def collect(directory):
    directory = Path(directory)
    return {str(path.relative_to(directory)): path.read_bytes()
            for path in sorted(directory.rglob("*")) if path.is_file()}


@app.function(image=image, cpu=8, memory=16384, timeout=1200, retries=0,
              max_containers=1, scaledown_window=2, serialized=True)
def cpu_build(request):
    candidate = "/opt/nextgen/c1" if request["track"] == "c1" else "/opt/c2/nextgen"
    out = Path("/tmp/nextgen-build")
    out.mkdir(exist_ok=False)
    records = [command(["nvcc", "--version"], timeout=15)]
    argv = ["python", "-u", candidate + "/build.py", "--arch", request["arch"], "--output", str(out)] + request.get("build_args", [])
    records.append(command(argv, timeout=1100, cwd=candidate))
    artifacts = {name: data for name, data in collect(out).items()
                 if not name.endswith(".o") and not Path(name).name.startswith(".ninja")}
    success = all(r["returncode"] == 0 for r in records)
    if success and not any(name.endswith(".so") for name in artifacts):
        success = False
        records.append({"command": ["check-built-extension"], "returncode": 1,
                        "stderr": "No .so found in the declared build directory"})
    blob = io.BytesIO()
    with tarfile.open(fileobj=blob, mode="w:gz") as archive:
        for name, data in artifacts.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return {"success": success, "records": records, "files": list(artifacts),
            "bundle": blob.getvalue(), "bundle_sha256": hashlib.sha256(blob.getvalue()).hexdigest()}


@app.function(image=image, gpu="B300", cpu=4, memory=16384, timeout=1200,
              retries=0, max_containers=1, scaledown_window=2, serialized=True)
def gpu_experiment(request, build_bundle):
    build = Path("/tmp/nextgen-build")
    build.mkdir(exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(build_bundle), mode="r:gz") as archive:
        for member in archive.getmembers():
            target = (build / member.name).resolve()
            if not target.is_relative_to(build) or not member.isfile():
                raise ValueError("Invalid build artifact path")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.extractfile(member).read())
    out = Path("/tmp/nextgen-artifacts")
    out.mkdir(exist_ok=False)
    candidate = "/opt/nextgen/c1" if request["track"] == "c1" else "/opt/c2/nextgen"
    records = []
    for argv in (["nvidia-smi"], ["nvcc", "--version"],
                 ["git", "-C", "/opt/FlashKDA", "rev-parse", "HEAD"],
                 ["git", "-C", "/opt/FlashKDA/cutlass", "rev-parse", "HEAD"]):
        records.append(command(argv, timeout=20))
    argv = ["python", "-u", candidate + "/run.py", "--mode", request["mode"],
            "--output", str(out), "--build-dir", str(build)] + request["extra_args"]
    limit = {"smoke": 240, "verify": 950, "bench": 700, "profile": 950}[request["mode"]]
    records.append(command(argv, timeout=limit, cwd="/opt/c2" if request["track"] == "c2" else candidate))
    (out / "remote_commands.json").write_text(json.dumps(records, indent=2) + "\n")
    for index, binary in enumerate(sorted(build.rglob("*.so"))):
        resources = command(["cuobjdump", "--dump-resource-usage", str(binary)], timeout=30)
        (out / f"binary-{index}-resources.json").write_text(json.dumps(resources, indent=2) + "\n")
        sass = command(["cuobjdump", "--dump-sass", str(binary)], timeout=60)
        (out / f"binary-{index}.sass").write_text(sass.get("stdout") or "")
        (out / f"binary-{index}-sass-command.json").write_text(json.dumps({k: v for k, v in sass.items() if k != "stdout"}, indent=2) + "\n")
    return {"success": all(r["returncode"] == 0 for r in records), "artifacts": collect(out)}


def save_state(status, **kwargs):
    timestamp = datetime.now(timezone.utc).isoformat()
    previous = json.loads((RUN_DIR / "state.json").read_text()) if (RUN_DIR / "state.json").exists() else {}
    value = {**previous, "status": status, "updated_at": timestamp, **kwargs}
    value.setdefault("started_at", timestamp)
    if status == "BUILD_RUNNING":
        value["build_call_id"] = kwargs.get("function_call_id")
    if status == "GPU_RUNNING":
        value["gpu_call_id"] = kwargs.get("function_call_id")
    with (RUN_DIR / "state_history.jsonl").open("a") as history:
        history.write(json.dumps(value) + "\n")
    (RUN_DIR / "state.json").write_text(json.dumps(value, indent=2) + "\n")


@app.local_entrypoint()
def main():
    if (RUN_DIR / "state.json").exists():
        raise RuntimeError("This run already has state. Inspect its live handle; do not blindly restart.")
    # Recheck frozen source bytes before any remote submission.
    for row in json.loads((RUN_DIR / "source_manifest.json").read_text()):
        if hashlib.sha256((RUN_DIR / "sources" / row["path"]).read_bytes()).hexdigest() != row["sha256"]:
            raise RuntimeError("Frozen source changed: " + row["path"])
    if REQUEST.get("reuse_build"):
        built = json.loads((RUN_DIR / "reused_build.json").read_text())
        built["bundle"] = (RUN_DIR / "reused_build.tar.gz").read_bytes()
        if hashlib.sha256(built["bundle"]).hexdigest() != REQUEST["reuse_build"]["bundle_sha256"]:
            raise RuntimeError("Reused build hash mismatch")
        save_state("BUILD_REUSED", reuse_build=REQUEST["reuse_build"])
    else:
        save_state("BUILD_SUBMITTING")
        call = cpu_build.spawn(REQUEST)
        save_state("BUILD_RUNNING", function_call_id=call.object_id)
        print(f"NEXTGEN_BUILD_CALL={call.object_id}", flush=True)
        built = call.get()
    (RUN_DIR / "build.json").write_text(json.dumps({k: v for k, v in built.items() if k != "bundle"}, indent=2) + "\n")
    (RUN_DIR / "build.tar.gz").write_bytes(built["bundle"])
    if not built["success"]:
        save_state("BUILD_FAILED")
        raise RuntimeError("CPU build failed; no GPU submitted. See build.json.")
    call = gpu_experiment.spawn(REQUEST, built["bundle"])
    save_state("GPU_RUNNING", function_call_id=call.object_id)
    print(f"NEXTGEN_GPU_CALL={call.object_id}", flush=True)
    result = call.get()
    for name, data in result["artifacts"].items():
        target = RUN_DIR / "artifacts" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    save_state("COMPLETE" if result["success"] else "GPU_FAILED", function_call_id=call.object_id)
    print(f"NEXTGEN_SAVED={RUN_DIR}", flush=True)
    if not result["success"]:
        raise RuntimeError("GPU subprocess failed; original output is persisted.")
