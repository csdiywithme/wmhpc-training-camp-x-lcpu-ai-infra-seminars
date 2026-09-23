"""Bounded rectangular GEMM and epilogue ablation on one B300."""
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import modal

FILES = ("base_image.py", "modal_runner.py", "compile_one.py", "benchmark.py",
         "kernels_advanced.py", "kernels_tuned.py", "profile_ncu.py",
         "kernels_release.py", "kernels_k128.py")
ROOT = Path("/opt/gemm_tuning")
if modal.is_local():
    from base_image import image
    HERE = Path(__file__).resolve().parent
    if os.environ.get("MODAL_PROFILE") != "simidawhu":
        raise RuntimeError("Set MODAL_PROFILE=simidawhu explicitly")
    for name in FILES:
        image = image.add_local_file(HERE / name, str(ROOT / name))
else:
    image = None
app = modal.App("mlc-b300-pipeline-refinement")

def command(argv, timeout, stream=False):
    import signal
    import subprocess
    import threading
    import time
    start = time.monotonic()
    proc = subprocess.Popen(argv, cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, start_new_session=True)
    lines = []
    def consume():
        for line in proc.stdout:
            lines.append(line)
            if stream:
                print(line, end="", flush=True)
    reader = threading.Thread(target=consume, daemon=True)
    reader.start()
    timed_out = False
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    reader.join(timeout=10)
    result = {"command": argv, "returncode": None if timed_out else proc.returncode,
              "stdout": "".join(lines), "seconds": time.monotonic()-start}
    if timed_out:
        result["error"] = "timeout"
    print("COMMAND", " ".join(argv[:7]), result["returncode"], round(result["seconds"], 2), flush=True)
    if result["returncode"] != 0 and not stream:
        print(result["stdout"][-5000:], flush=True)
    return result

def collect(path):
    return {str(p.relative_to(path)): p.read_bytes() for p in sorted(path.rglob("*")) if p.is_file()}

def pack(files):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return stream.getvalue()

@app.function(image=image, cpu=16, memory=32768, timeout=1200, retries=0,
              max_containers=1, scaledown_window=2)
def cpu_build(request):
    from concurrent.futures import ThreadPoolExecutor
    import inspect
    from tvm.backend.cuda.lang.tile_scheduler import ClusterPersistentScheduler2D
    out = Path("/tmp/tuning-build")
    out.mkdir(exist_ok=True)
    (out / "request.json").write_text(json.dumps(request))
    (out / "scheduler.py").write_text(inspect.getsource(ClusterPersistentScheduler2D))
    records = [command(["nvcc", "--version"], 20)]
    jobs = [(c["name"], v) for c in request["cases"] for v in c["variants"]]
    def build(job):
        case, variant = job
        return command(["python", "-u", "compile_one.py", "--request", str(out / "request.json"),
                        "--case", case, "--variant", variant, "--output", str(out)], 180)
    with ThreadPoolExecutor(max_workers=4) as pool:
        records.extend(pool.map(build, jobs))
    return {"success": all(r["returncode"] == 0 for r in records), "records": records,
            "bundle": pack(collect(out)), "sources": {name: (ROOT/name).read_bytes() for name in FILES}}

@app.function(image=image, gpu="B300", cpu=4, memory=16384, timeout=2100, retries=0,
              max_containers=1, scaledown_window=2)
def gpu_run(request, bundle):
    build, out = Path("/tmp/tuning-build"), Path("/tmp/tuning-results")
    build.mkdir(exist_ok=True)
    out.mkdir(exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(bundle), mode="r:gz") as tar:
        for member in tar.getmembers():
            p = (build/member.name).resolve()
            if not p.is_relative_to(build) or not member.isfile():
                raise ValueError("Invalid artifact path")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(tar.extractfile(member).read())
    (out/"request.json").write_text(json.dumps(request, indent=2))
    records = [command(["nvidia-smi"], 20, stream=True)]
    records.append(command(["python", "-u", "benchmark.py", "--request", str(out/"request.json"),
                            "--build", str(build), "--output", str(out)], 1400, stream=True))
    if records[-1]["returncode"] == 0 and request.get("collect_ncu"):
        records.append(command(["python", "-u", "profile_ncu.py", "--request", str(out/"request.json"),
                                "--build", str(build), "--output", str(out/"ncu"),
                                "--results", str(out/"results.json")], 650, stream=True))
    return {"records": records, "artifacts": collect(out)}

@app.local_entrypoint()
def main(request_file: str = "", build_only: bool = False, reuse_build: str = ""):
    source = Path(request_file).resolve() if request_file else HERE/"request.json"
    request = json.loads(source.read_text())
    out = HERE/"runs"/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True)
    (out/"request.json").write_text(json.dumps(request, indent=2)+"\n")
    manifest = {}
    for name in FILES:
        data = (HERE/name).read_bytes()
        p = out/"sources"/name
        p.parent.mkdir(exist_ok=True)
        p.write_bytes(data)
        manifest[name] = hashlib.sha256(data).hexdigest()
    (out/"source_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print("RUN_DIRECTORY", out, flush=True)
    try:
        if reuse_build:
            previous = Path(reuse_build).resolve()
            old = json.loads((previous/"request.json").read_text())
            for key in ("cases", "sm_count", "arch"):
                if old[key] != request[key]:
                    raise ValueError("Reused build differs: " + key)
            prev_hash = json.loads((previous/"source_manifest.json").read_text())
            for name in ("base_image.py", "compile_one.py", "kernels_advanced.py", "kernels_tuned.py"):
                if prev_hash[name] != manifest[name]:
                    raise ValueError("Reused build source differs: " + name)
            bundle = (previous/"build.tar.gz").read_bytes()
            (out/"reused_build.txt").write_text(str(previous)+"\n")
        else:
            built = cpu_build.remote(request)
            bundle = built.pop("bundle")
            for name, data in built.pop("sources").items():
                if hashlib.sha256(data).hexdigest() != manifest[name]:
                    raise RuntimeError("Remote source differs: " + name)
            (out/"build.json").write_text(json.dumps(built, indent=2)+"\n")
            (out/"build.tar.gz").write_bytes(bundle)
            if not built["success"]:
                raise RuntimeError("CPU build failed; no GPU launched")
        (out/"build.tar.gz").write_bytes(bundle)
        if not build_only:
            result = gpu_run.remote(request, bundle)
            for name, data in result.pop("artifacts").items():
                p = out/"artifacts"/name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(data)
            (out/"gpu.json").write_text(json.dumps(result, indent=2)+"\n")
            if any(r["returncode"] != 0 for r in result["records"]):
                raise RuntimeError("GPU validation/measurement failed; partial results preserved")
        print("SAVED", out, flush=True)
    except BaseException as exc:
        (out/"error.txt").write_text(str(exc)+"\n")
        raise
