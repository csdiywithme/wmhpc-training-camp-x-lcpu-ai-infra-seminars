"""CPU-compile the nine tutorial variants, then measure on one Modal B300.

Only this experiment's explicitly listed source files are uploaded. Each run
freezes the exact sources and preserves build/GPU diagnostics, including errors.
"""
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import modal

FILES = ("base_image.py", "modal_runner.py", "compile_one.py", "benchmark.py",
         "kernels_basics.py", "kernels_advanced.py")
ROOT = Path("/opt/mlc_gemm")

if modal.is_local():
    from base_image import image
    HERE = Path(__file__).resolve().parent
    if os.environ.get("MODAL_PROFILE") != "simidawhu":
        raise RuntimeError("Set MODAL_PROFILE=simidawhu explicitly")
    for name in FILES:
        image = image.add_local_file(HERE / name, str(ROOT / name))
else:
    image = None

app = modal.App("mlc-nine-gemm-b300")

def command(argv, timeout):
    import subprocess
    import time
    start = time.monotonic()
    try:
        p = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
        result = dict(command=argv, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
    except subprocess.TimeoutExpired as e:
        decode = lambda x: x.decode(errors="replace") if isinstance(x, bytes) else (x or "")
        result = dict(command=argv, returncode=None, error="timeout",
                      stdout=decode(e.stdout), stderr=decode(e.stderr))
    result["seconds"] = time.monotonic() - start
    print(json.dumps(result), flush=True)
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

@app.function(image=image, cpu=8, memory=16384, timeout=1800, retries=0,
              max_containers=1, scaledown_window=2)
def cpu_build(request):
    out = Path("/tmp/gemm-build")
    out.mkdir(exist_ok=True)
    records = [command(["nvcc", "--version"], 15)]
    for v in request["versions"]:
        records.append(command(["python", "-u", "compile_one.py", "--version", str(v),
                                "--size", str(request["size"]), "--sm-count", str(request["sm_count"]),
                                "--arch", "sm_103a", "--output", str(out)], 150))
    files = collect(out)
    return {"success": all(r["returncode"] == 0 for r in records),
            "records": records, "bundle": pack(files), "files": list(files),
            "sources": {name: (ROOT / name).read_bytes() for name in FILES}}

@app.function(image=image, gpu="B300", cpu=4, memory=16384, timeout=1500, retries=0,
              max_containers=1, scaledown_window=2)
def gpu_run(request, bundle):
    build = Path("/tmp/gemm-build")
    out = Path("/tmp/gemm-results")
    build.mkdir(exist_ok=True)
    out.mkdir(exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(bundle), mode="r:gz") as tar:
        for member in tar.getmembers():
            p = (build / member.name).resolve()
            if not p.is_relative_to(build) or not member.isfile():
                raise ValueError("Unsafe artifact path")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(tar.extractfile(member).read())
    records = [command(["nvidia-smi"], 15)]
    (out / "request.json").write_text(json.dumps(request, indent=2))
    records.append(command(["python", "-u", "benchmark.py", "--request", str(out / "request.json"),
                            "--build", str(build), "--output", str(out)], 1400))
    return {"records": records, "artifacts": collect(out)}

@app.local_entrypoint()
def main(size: int = 4096, versions: str = "1,2,3,4,5,6,7,8,9", rounds: int = 5,
         repeat: int = 200, mode: str = "bench", build_only: bool = False,
         reuse_build: str = ""):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = HERE / "runs" / stamp
    out.mkdir(parents=True)
    request = {"size": size, "versions": [int(v) for v in versions.split(",")],
               "rounds": rounds, "repeat": repeat, "mode": mode, "sm_count": 148,
               "arch": "sm_103a", "gpu": "B300", "profile": "simidawhu"}
    (out / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    for name in FILES:
        p = out / "sources" / name
        p.parent.mkdir(exist_ok=True)
        p.write_bytes((HERE / name).read_bytes())
    manifest = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in FILES}
    (out / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("RUN_DIRECTORY", out, flush=True)
    try:
        if reuse_build:
            previous = Path(reuse_build).resolve()
            old_request = json.loads((previous / "request.json").read_text())
            for key in ("size", "versions", "sm_count", "arch"):
                if request[key] != old_request[key]:
                    raise ValueError(f"Reused build differs in {key}")
            old_manifest = json.loads((previous / "source_manifest.json").read_text())
            for name in ("base_image.py", "compile_one.py", "kernels_basics.py", "kernels_advanced.py"):
                if manifest[name] != old_manifest[name]:
                    raise ValueError(f"Reused build source differs: {name}")
            bundle = (previous / "build.tar.gz").read_bytes()
            (out / "reused_build.txt").write_text(str(previous) + "\n")
        else:
            built = cpu_build.remote(request)
            bundle = built.pop("bundle")
            actual_sources = built.pop("sources")
            for name, data in actual_sources.items():
                if hashlib.sha256(data).hexdigest() != manifest[name]:
                    raise RuntimeError(f"Remote source mismatch: {name}")
            (out / "build.json").write_text(json.dumps(built, indent=2) + "\n")
            (out / "build.tar.gz").write_bytes(bundle)
            if not built["success"]:
                raise RuntimeError("CPU compilation failed; no GPU experiment started")
        (out / "build.tar.gz").write_bytes(bundle)
        if not build_only:
            result = gpu_run.remote(request, bundle)
            for name, data in result.pop("artifacts").items():
                p = out / "artifacts" / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(data)
            (out / "gpu.json").write_text(json.dumps(result, indent=2) + "\n")
            if any(r["returncode"] != 0 for r in result["records"]):
                raise RuntimeError("GPU experiment failed; partial results preserved")
        print("SAVED", out, flush=True)
    except Exception as exc:
        (out / "error.txt").write_text(str(exc) + "\n")
        raise
