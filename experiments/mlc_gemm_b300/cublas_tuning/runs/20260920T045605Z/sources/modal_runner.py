"""Run only T1--T3, durably checkpointing one B300 invocation to a Modal Volume.

Launch with ``modal run --detach .../modal_runner.py --run-dir <absolute-path>``.
Use the same command with ``--retrieve-only`` to read saved outputs without
launching or resuming a GPU experiment. A run directory is never launched twice.
"""

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import time
import uuid

import modal


FILES = ("base_image.py", "modal_runner.py", "benchmark.py", "native.cu")
ROOT = Path("/opt/cublas_tuning")
VOLUME_ROOT = Path("/results")
VOLUME_NAME = "mlc-b300-cublas-tuning-results"
HERE = Path(__file__).resolve().parent

if modal.is_local():
    if os.environ.get("MODAL_PROFILE") != "simidawhu":
        raise RuntimeError("Set MODAL_PROFILE=simidawhu explicitly")
    spec = importlib.util.spec_from_file_location("cublas_tuning_image", HERE / "base_image.py")
    image_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(image_module)
    image = image_module.image
    for name in FILES:
        image = image.add_local_file(HERE / name, str(ROOT / name))
else:
    image = None

app = modal.App("mlc-b300-cublas-tuning")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def run_path(run_id):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", run_id):
        raise ValueError("Invalid run_id")
    return VOLUME_ROOT / run_id


def source_manifest(directory):
    return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in FILES}


def pinned_cublas(expected):
    """Locate the pip wheel rather than accidentally using the CUDA image's library."""
    from importlib.metadata import distribution

    package = distribution("nvidia-cublas")
    if package.version != expected:
        raise RuntimeError(f"cuBLAS wheel mismatch: {package.version} != {expected}")
    matches = [Path(package.locate_file(file)).resolve() for file in package.files
               if str(file).endswith("/libcublas.so.13")]
    if len(matches) != 1:
        raise RuntimeError(f"Cannot uniquely locate pip libcublas.so.13: {matches}")
    directory = matches[0].parent
    if not (directory / "libcublasLt.so.13").is_file():
        raise RuntimeError("Pinned cuBLASLt shared library is missing")
    return directory


def command(argv, timeout=120, env=None):
    import subprocess

    started = time.monotonic()
    try:
        proc = subprocess.run(argv, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, timeout=timeout)
        record = {"command": argv, "returncode": proc.returncode, "stdout": proc.stdout}
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        record = {"command": argv, "returncode": None, "stdout": output, "error": "timeout"}
    record["seconds"] = time.monotonic() - started
    print("COMMAND", argv[0], record["returncode"], round(record["seconds"], 2), flush=True)
    return record


def collect(directory):
    return {str(path.relative_to(directory)): path.read_bytes()
            for path in sorted(directory.rglob("*")) if path.is_file()}


@app.function(image=image, cpu=4, memory=16384, timeout=600, retries=0,
              max_containers=1, scaledown_window=2, volumes={str(VOLUME_ROOT): volume})
def cpu_build(request, run_id, manifest):
    """Compile without allocating a GPU; retain the compiler logs and .so durably."""
    volume.reload()
    destination = run_path(run_id)
    if destination.exists():
        raise RuntimeError("Run already exists on the Volume; retrieve it instead of rerunning")
    destination.mkdir(parents=True)
    build = destination / "build"
    sources = destination / "sources"
    build.mkdir()
    sources.mkdir()
    records = []
    status = {"run_id": run_id, "state": "building", "started_at": now()}
    try:
        remote_manifest = source_manifest(ROOT)
        if remote_manifest != manifest:
            raise RuntimeError("Remote source hashes differ from the local snapshot")
        for name in FILES:
            (sources / name).write_bytes((ROOT / name).read_bytes())
        atomic_json(destination / "request.json", request)
        atomic_json(destination / "source_manifest.json", manifest)
        atomic_json(destination / "status.json", status)
        volume.commit()

        library_dir = pinned_cublas(request["expected_cublas_package"])
        # Wheels contain versioned sonames, without the unversioned linker names.
        # The temporary links select those exact libraries for -lcublas{Lt}.
        links = Path("/tmp/cublas-pinned-link")
        links.mkdir(exist_ok=True)
        for name in ("cublas", "cublasLt"):
            link = links / f"lib{name}.so"
            link.unlink(missing_ok=True)
            link.symlink_to(library_dir / f"lib{name}.so.13")
        environment = os.environ.copy()
        environment["LD_LIBRARY_PATH"] = str(library_dir) + ":" + environment.get("LD_LIBRARY_PATH", "")
        shared_object = build / "libgemmbench.so"
        records.append(command(["nvcc", "--version"], timeout=20, env=environment))
        records.append(command([
            "nvcc", "--shared", "-Xcompiler", "-fPIC", "-O3", "-std=c++17",
            "-I/usr/local/cuda/include", str(ROOT / "native.cu"),
            "-L" + str(links), "-L/usr/local/cuda/lib64",
            "-Xlinker", "-rpath", "-Xlinker", str(library_dir),
            "-lcublasLt", "-lcublas", "-lcudart", "-o", str(shared_object),
        ], timeout=480, env=environment))
        if any(record["returncode"] != 0 for record in records):
            raise RuntimeError("CPU native-library compilation failed; no GPU was launched")
        records.append(command(["ldd", str(shared_object)], timeout=20, env=environment))
        if records[-1]["returncode"] != 0 or "not found" in records[-1]["stdout"]:
            raise RuntimeError("Native-library dependency resolution failed")
        for name in ("libcublas.so.13", "libcublasLt.so.13"):
            line = next((line for line in records[-1]["stdout"].splitlines() if name in line), "")
            if str(library_dir) not in line:
                raise RuntimeError(f"Native library did not resolve {name} to the pinned wheel: {line}")
        metadata = {"success": True, "records": records,
                    "cublas_library_directory": str(library_dir),
                    "shared_object_sha256": hashlib.sha256(shared_object.read_bytes()).hexdigest()}
        atomic_json(build / "build.json", metadata)
        status.update(state="built", build_finished_at=now())
        atomic_json(destination / "status.json", status)
        volume.commit()
        return {"success": True, "files": collect(destination)}
    except BaseException as exc:
        atomic_json(build / "build.json", {"success": False, "records": records, "error": repr(exc)})
        status.update(state="build_failed", finished_at=now(), error=repr(exc))
        atomic_json(destination / "status.json", status)
        volume.commit()
        return {"success": False, "files": collect(destination)}


class CheckpointConsole:
    """Tee Python logs, closing the Volume file after every write before commits."""

    def __init__(self, original, path):
        self.original, self.path = original, path

    def write(self, value):
        self.original.write(value)
        self.original.flush()
        with self.path.open("a") as handle:
            handle.write(value)
        return len(value)

    def flush(self):
        self.original.flush()

    def isatty(self):
        return False


@app.function(image=image, gpu="B300", cpu=4, memory=32768, timeout=1800, retries=0,
              max_containers=1, scaledown_window=2, volumes={str(VOLUME_ROOT): volume})
def gpu_run(request, run_id, manifest):
    """One invocation only; never retries completed cases or resumes a prior run."""
    import contextlib
    import ctypes
    import sys
    import traceback

    volume.reload()
    destination = run_path(run_id)
    status_path = destination / "status.json"
    status = json.loads(status_path.read_text())
    if status["state"] != "built":
        raise RuntimeError(f"Refusing repeated GPU launch from state {status['state']!r}")
    if json.loads((destination / "request.json").read_text()) != request:
        raise RuntimeError("GPU request differs from the immutable build request")
    if json.loads((destination / "source_manifest.json").read_text()) != manifest or source_manifest(ROOT) != manifest:
        raise RuntimeError("GPU source snapshot differs from the build")
    artifacts = destination / "artifacts"
    artifacts.mkdir(exist_ok=True)
    status.update(state="running", gpu_started_at=now())
    atomic_json(status_path, status)
    volume.commit()
    last_commit = time.monotonic()

    def checkpoint_callback(force=False):
        nonlocal last_commit
        if force or time.monotonic() - last_commit >= 20:
            status["last_checkpoint_at"] = now()
            atomic_json(status_path, status)
            volume.commit()
            last_commit = time.monotonic()

    try:
        library_dir = pinned_cublas(request["expected_cublas_package"])
        os.environ["LD_LIBRARY_PATH"] = str(library_dir) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
        # Preload before importing torch: this also makes the selection explicit
        # when the dynamic loader captured LD_LIBRARY_PATH at process startup.
        loaded_libraries = [ctypes.CDLL(str(library_dir / name), mode=ctypes.RTLD_GLOBAL)
                            for name in ("libcublasLt.so.13", "libcublas.so.13")]
        library_path = destination / "build" / "libgemmbench.so"
        build = json.loads((destination / "build" / "build.json").read_text())
        if hashlib.sha256(library_path.read_bytes()).hexdigest() != build["shared_object_sha256"]:
            raise RuntimeError("Compiled shared-library checksum changed")
        environment_records = [command(["nvidia-smi"], timeout=20),
                               command(["ldd", str(library_path)], timeout=20)]
        atomic_json(artifacts / "runner_environment.json", {
            "records": environment_records, "cublas_library_directory": str(library_dir),
            "ld_library_path": os.environ["LD_LIBRARY_PATH"],
            "source_manifest": manifest, "shared_object_sha256": build["shared_object_sha256"],
        })
        checkpoint_callback(force=True)
        sys.path.insert(0, str(ROOT))
        console = CheckpointConsole(sys.stdout, artifacts / "console.log")
        errors = CheckpointConsole(sys.stderr, artifacts / "console.log")
        with contextlib.redirect_stdout(console), contextlib.redirect_stderr(errors):
            import benchmark
            result = benchmark.run(request, artifacts, library_path, checkpoint_callback=checkpoint_callback)
        status.update(state="complete", finished_at=now())
        atomic_json(status_path, status)
        volume.commit()
        # Results remain durable even if this small final RPC return is lost.
        return {"success": True, "run_id": run_id, "volume": VOLUME_NAME,
                "result_path": str(artifacts / "results.json"), "status": status}
    except BaseException as exc:
        atomic_json(artifacts / "runner_error.json", {"error": repr(exc), "traceback": traceback.format_exc()})
        status.update(state="failed", finished_at=now(), error=repr(exc))
        atomic_json(status_path, status)
        volume.commit()
        raise


@app.function(image=image, cpu=1, memory=2048, timeout=180, retries=0,
              max_containers=1, scaledown_window=2, volumes={str(VOLUME_ROOT): volume})
def retrieve(run_id):
    """Read durable files; this function cannot launch a GPU experiment."""
    volume.reload()
    destination = run_path(run_id)
    if not destination.is_dir():
        raise FileNotFoundError(f"No durable run {run_id!r}")
    return collect(destination)


def save_files(destination, files):
    for name, data in files.items():
        target = (destination / name).resolve()
        if not target.is_relative_to(destination.resolve()):
            raise ValueError("Invalid returned artifact path")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


@app.local_entrypoint()
def main(request_file: str = "", run_dir: str = "", build_only: bool = False,
         retrieve_only: bool = False):
    if retrieve_only:
        if not run_dir:
            raise ValueError("--retrieve-only requires the existing --run-dir")
        destination = Path(run_dir).resolve()
        invocation = json.loads((destination / "invocation.json").read_text())
        save_files(destination, retrieve.remote(invocation["run_id"]))
        print("RETRIEVED", destination, flush=True)
        print((destination / "status.json").read_text(), flush=True)
        return

    request_path = Path(request_file).resolve() if request_file else HERE / "request.json"
    request = json.loads(request_path.read_text())
    if set(request.get("scope", [])) != {"T1", "T2", "T3"} or len(request.get("cases", [])) != 7:
        raise ValueError("This runner is restricted to the authorized seven T1--T3 cases")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    destination = Path(run_dir).resolve() if run_dir else HERE / "runs" / run_id
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "invocation.json").exists():
        raise RuntimeError("This local run was already launched; use --retrieve-only")
    manifest = source_manifest(HERE)
    sources = destination / "sources"
    sources.mkdir(exist_ok=True)
    for name in FILES:
        (sources / name).write_bytes((HERE / name).read_bytes())
    atomic_json(destination / "request.json", request)
    atomic_json(destination / "source_manifest.json", manifest)
    invocation = {"run_id": run_id, "created_at": now(), "volume": VOLUME_NAME,
                  "volume_path": "/" + run_id, "app_name": "mlc-b300-cublas-tuning",
                  "build_only": build_only}
    atomic_json(destination / "invocation.json", invocation)
    print("RUN_DIRECTORY", destination, flush=True)
    print("DURABLE_VOLUME", VOLUME_NAME, "/" + run_id, flush=True)
    try:
        built = cpu_build.remote(request, run_id, manifest)
        save_files(destination, built["files"])
        if not built["success"]:
            raise RuntimeError("CPU build failed; see build/build.json; no GPU was launched")
        if build_only:
            print("BUILD_ONLY_SAVED", destination, flush=True)
            return
        call = gpu_run.spawn(request, run_id, manifest)
        invocation.update(gpu_call_id=call.object_id, gpu_spawned_at=now())
        atomic_json(destination / "invocation.json", invocation)
        print("GPU_CALL_ID", call.object_id, flush=True)
        deadline = time.monotonic() + 1920
        while True:
            try:
                result = call.get(timeout=60)
                atomic_json(destination / "gpu_return.json", result)
                break
            except (TimeoutError, modal.exception.TimeoutError):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Stopped waiting; retrieve Volume outputs without relaunching the GPU")
                print("WAITING_FOR_GPU", run_id, flush=True)
        save_files(destination, retrieve.remote(run_id))
        print("SAVED", destination, flush=True)
    except BaseException as exc:
        atomic_json(destination / "local_error.json", {"error": repr(exc), "at": now(),
                    "recovery": "Use --retrieve-only with this same --run-dir; do not rerun cases"})
        # A failed client return is not permission to execute the experiment again.
        print("RESULTS_PERSIST_ON_VOLUME", VOLUME_NAME, "/" + run_id, flush=True)
        raise
