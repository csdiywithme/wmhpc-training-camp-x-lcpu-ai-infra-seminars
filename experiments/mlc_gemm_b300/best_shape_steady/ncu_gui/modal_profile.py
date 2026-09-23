"""One NCU capture of the saved wide 2048x9472x8192 kernel.

No benchmark sweep or automatic GPU retry. Reports are committed to a Volume;
download them with `modal volume get` into an existing local directory.
"""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import uuid

import modal

HERE = Path(__file__).resolve().parent
ROOT = Path("/opt/wide_ncu_gui")
VOLUME_ROOT = Path("/results")
VOLUME_NAME = "mlc-b300-wide-ncu-results"
OLD_RUN = "20260920T051356Z"
STEM = "aligned148_k8192__wide"
SECTIONS = ["SpeedOfLight", "SpeedOfLight_RooflineChart", "ComputeWorkloadAnalysis",
            "MemoryWorkloadAnalysis", "MemoryWorkloadAnalysis_Chart", "MemoryWorkloadAnalysis_Tables",
            "SchedulerStats", "WarpStateStats", "InstructionStats", "LaunchStats", "Occupancy",
            "WorkloadDistribution", "SourceCounters"]

if modal.is_local():
    if os.environ.get("MODAL_PROFILE") != "simidawhu":
        raise RuntimeError("Set MODAL_PROFILE=simidawhu explicitly")
    old = HERE.parent / "runs" / OLD_RUN
    spec = importlib.util.spec_from_file_location("wide_ncu_base", HERE.parent / "base_image.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    image = module.image.apt_install("cuda-nsight-compute-13-1")
    INPUTS = {f"input/{STEM}{suffix}": old / "build" / (STEM + suffix)
              for suffix in (".so", ".cu", ".tirx.py", ".compile.json")}
    INPUTS["input/kernels_tuned.py"] = old / "sources/kernels_tuned.py"
    INPUTS["profile_target.py"] = HERE / "profile_target.py"
    INPUTS["modal_profile.py"] = HERE / "modal_profile.py"
    for name, path in INPUTS.items():
        image = image.add_local_file(path, str(ROOT / name))
else:
    image = None

app = modal.App("mlc-b300-wide-ncu-gui")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def command(argv, output, timeout):
    import signal
    import subprocess
    import threading
    import time

    started = time.monotonic()
    environment = os.environ.copy()
    environment["NV_COMPUTE_PROFILER_DISABLE_STOCK_FILE_DEPLOYMENT"] = "1"
    process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, env=environment, start_new_session=True)
    def consume():
        with output.open("w") as stream:
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                if output.name == "profile.log":
                    print(line, end="", flush=True)
    reader = threading.Thread(target=consume, daemon=True)
    reader.start()
    timed_out = False
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)
    reader.join(timeout=10)
    result = {"argv": list(map(str, argv)), "returncode": process.returncode,
              "timed_out": timed_out, "seconds": time.monotonic() - started,
              "log": output.name}
    print("COMMAND_COMPLETE", output.name, process.returncode, flush=True)
    return result


@app.function(image=image, gpu="B300", cpu=4, memory=16384, timeout=600,
              retries=0, max_containers=1, scaledown_window=2,
              volumes={str(VOLUME_ROOT): volume})
def capture(run_id, manifest, collection):
    import sys
    import traceback

    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("Invalid run ID")
    if collection not in ("full", "sections"):
        raise ValueError("Invalid collection mode")
    report_name = f"wide_M2048_N9472_K8192_{collection}.ncu-rep"
    volume.reload()
    out = VOLUME_ROOT / run_id
    if out.exists():
        raise RuntimeError("Run already exists; retrieve it without profiling again")
    out.mkdir()
    status = {"state": "preparing", "started_at": stamp(), "shape": [2048, 9472, 8192],
              "variant": "wide", "source_benchmark_run": OLD_RUN,
              "source_manifest": manifest, "collection": collection, "commands": []}
    def checkpoint():
        save_json(out / "status.json", status)
        volume.commit()
    checkpoint()
    try:
        for name, digest in manifest.items():
            source = ROOT / name
            if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"Input checksum mismatch: {name}")
            target = out / name
            target.parent.mkdir(exist_ok=True, parents=True)
            shutil.copyfile(source, target)
        build = json.loads((out / "input" / (STEM + ".compile.json")).read_text())
        if build["case"]["shape"] != [2048, 9472, 8192] or build["variant"] != "wide":
            raise RuntimeError("Unexpected saved kernel")
        for suffix in (".so", ".cu"):
            name = STEM + suffix
            if manifest["input/" + name] != build["sha256"][name]:
                raise RuntimeError("Saved benchmark artifact checksum differs")
        ncu = shutil.which("ncu") or "/usr/local/cuda/bin/ncu"
        for args, log in (([ncu, "--version"], "ncu_version.txt"),
                          ([ncu, "--list-sets"], "ncu_sets.txt"),
                          (["nvidia-smi"], "nvidia_smi.txt")):
            record = command(args, out / log, 30)
            status["commands"].append(record)
            if record["returncode"] != 0:
                raise RuntimeError(f"Preflight failed: {log}")
        status["state"] = "profiling"
        checkpoint()
        selection = ["--set", "full"] if collection == "full" else [
            token for section in SECTIONS for token in ("--section", section)] + [
            "--metrics", "sm__cycles_active.avg.pct_of_peak_sustained_elapsed,"
            "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,"
            "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"]
        argv = [ncu] + selection + ["--target-processes", "all", "--nvtx",
                "--nvtx-include", "wide_2048x9472x8192/", "--kernel-name-base", "function",
                "--kernel-name", "regex:^kernel_kernel$", "--launch-count", "1",
                "--replay-mode", "kernel", "--cache-control", "none", "--clock-control", "none",
                "--import-source", "yes", "--export", str(out / report_name),
                sys.executable, "-u", str(out / "profile_target.py"),
                "--library", str(out / "input" / (STEM + ".so")), "--output", str(out)]
        record = command(argv, out / "profile.log", 420)
        status["commands"].append(record)
        checkpoint()
        report = out / report_name
        if record["returncode"] != 0 or record["timed_out"] or not report.is_file():
            raise RuntimeError("NCU capture failed; inspect profile.log")
        for page, log, extra in (("details", "details.txt", []),
                                  ("raw", "metrics.csv", ["--csv", "--print-units", "base"]),
                                  ("session", "session.txt", [])):
            record = command([ncu, "--import", str(report), "--page", page] + extra, out / log, 30)
            status["commands"].append(record)
            if record["returncode"] != 0:
                raise RuntimeError(f"Report import failed: {page}")
        import csv
        import io
        import math
        text = (out / "metrics.csv").read_text().splitlines()
        start = next(i for i, line in enumerate(text) if line.startswith('"ID",'))
        rows = csv.DictReader(io.StringIO("\n".join(text[start:])))
        next(rows)  # Unit row
        measured = [row for row in rows if row.get("ID")]
        if len(measured) != 1:
            raise RuntimeError(f"Expected one target kernel, got {len(measured)}")
        required = ["gpu__time_duration.sum", "sm__cycles_elapsed.avg",
                    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                    "dram__cycles_elapsed.avg", "dram__bytes_read.sum",
                    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed"]
        valid = {name: math.isfinite(float(measured[0].get(name, "nan").replace(",", "")))
                 for name in required}
        status["counter_validation"] = valid
        if not all(valid.values()):
            raise RuntimeError("Required hardware counters are unavailable/NaN; report is partial")
        status.update(state="complete", finished_at=stamp(), report=report_name,
                      report_bytes=report.stat().st_size,
                      report_sha256=hashlib.sha256(report.read_bytes()).hexdigest())
        checkpoint()
        return {"success": True, "run_id": run_id, "volume": VOLUME_NAME,
                "report": report_name, "bytes": status["report_bytes"],
                "sha256": status["report_sha256"]}
    except BaseException as exc:
        status.update(state="failed", finished_at=stamp(), error=repr(exc), traceback=traceback.format_exc())
        checkpoint()
        raise


@app.local_entrypoint()
def main(run_dir: str, collection: str = "sections"):
    destination = Path(run_dir).resolve()
    destination.mkdir(exist_ok=True, parents=True)
    invocation = destination / "invocation.json"
    if invocation.exists():
        raise RuntimeError("Already launched; retrieve Volume files instead of rerunning")
    manifest = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in INPUTS.items()}
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    info = {"run_id": run_id, "volume": VOLUME_NAME, "created_at": stamp(),
            "manifest": manifest, "collection": collection}
    save_json(invocation, info)
    call = capture.spawn(run_id, manifest, collection)
    info["gpu_call_id"] = call.object_id
    save_json(invocation, info)
    print("GPU_CALL", json.dumps(info), flush=True)
    # Only the small completion record returns over RPC. Files are on the Volume.
    result = call.get()
    save_json(destination / "gpu_return.json", result)
    print("CAPTURE_COMPLETE", json.dumps(result), flush=True)
