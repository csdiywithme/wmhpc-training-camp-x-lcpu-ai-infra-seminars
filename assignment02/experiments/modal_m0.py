"""Run the unmodified M0 example once on one Modal B300.

From the repository root, in an existing environment with Modal configured:
    python -m modal run assignment02/experiments/modal_m0.py

The image builds both binaries on CPU. Only the three INPUT_FILES below are
uploaded as coursework inputs; no other exercises or team files are mounted.
Results are saved under experiments/results/m0-<UTC timestamp>/.
This runner records observations; the written M0 explanations remain your work.
"""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import modal


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parent.parent
CUDA_DIR = HERE.parent / "cuda"
BASE_IMAGE = "nvidia/cuda:13.1.0-devel-ubuntu24.04"
REMOTE_CUDA = Path("/opt/m0/cuda")
REMOTE_ARTIFACTS = Path("/opt/m0/artifacts")
INPUT_FILES = ("common.h", "Makefile", "m0_env/01_first_mma.cu")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def run_command(argv, *, cwd=None, timeout=10):
    """Capture failures as data, including partial output on timeouts."""
    started = time.monotonic()
    record = {"command": argv, "cwd": str(cwd) if cwd else None,
              "started_at_utc": utc_now(), "timeout_seconds": timeout}
    try:
        process = subprocess.run(
            argv, cwd=cwd, capture_output=True, text=True, timeout=timeout,
        )
        record.update(returncode=process.returncode,
                      stdout=process.stdout, stderr=process.stderr)
    except subprocess.TimeoutExpired as exc:
        def as_text(value):
            if isinstance(value, bytes):
                return value.decode("utf-8", errors="replace")
            return value or ""
        record.update(returncode=None, error="timeout",
                      stdout=as_text(exc.stdout), stderr=as_text(exc.stderr))
    except OSError as exc:
        record.update(returncode=None, error=str(exc), stdout="", stderr="")
    record["elapsed_seconds"] = time.monotonic() - started
    return record


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def distribution_version_or_none(name):
    # Modal injects its runtime into the container without necessarily installing
    # package distribution metadata. Optional metadata must not discard results.
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def build_m0():
    """Image build step: compile and inspect only; no GPU requested."""
    REMOTE_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    records = []

    def checked(argv):
        record = run_command(argv, cwd=REMOTE_CUDA, timeout=90)
        records.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if record["returncode"] != 0:
            raise RuntimeError(f"M0 image build failed: {argv}")

    for argv in (["nvcc", "--version"], ["cuobjdump", "--version"],
                 ["g++", "--version"], ["make", "--version"]):
        checked(argv)

    # Use the course Makefile verbatim, moving each product before the next
    # architecture is built. -B prevents a prior architecture being reused.
    target = "bin/m0_env/01_first_mma"
    for arch in ("100f", "120a"):
        checked(["make", "-B", f"ARCH={arch}", target])
        binary = REMOTE_ARTIFACTS / f"first_mma_{arch}"
        (REMOTE_CUDA / target).rename(binary)
        # An empty PTX listing is itself an observation, so retain its result
        # without converting it into an image-build failure.
        for option in ("--list-elf", "--list-ptx"):
            records.append(run_command(["cuobjdump", option, str(binary)]))

    checked(["make", "-B", "ARCH=100f", "ptx/m0_env/01_first_mma"])
    (REMOTE_CUDA / "m0_env/01_first_mma.ptx").rename(
        REMOTE_ARTIFACTS / "first_mma_100f.ptx"
    )
    manifest = {
        "built_at_utc": utc_now(), "base_image": BASE_IMAGE,
        "python": platform.python_version(), "platform": platform.platform(),
        "commands": records,
        "input_sha256": {
            name: sha256((REMOTE_CUDA / name).read_bytes()) for name in INPUT_FILES
        },
        "artifact_sha256": {
            path.name: sha256(path.read_bytes())
            for path in sorted(REMOTE_ARTIFACTS.iterdir()) if path.is_file()
        },
    }
    (REMOTE_ARTIFACTS / "build.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )


# copy=True includes these files at image-build time. build_m0 has no gpu=.
# https://modal.com/docs/guide/images#add-local-files-with-add_local_dir-and-add_local_file
image = (
    modal.Image.from_registry(BASE_IMAGE, add_python="3.11")
    .entrypoint([])
    .apt_install("build-essential")
)
for input_name in INPUT_FILES:
    image = image.add_local_file(
        CUDA_DIR / input_name, str(REMOTE_CUDA / input_name), copy=True,
    )
image = image.run_function(build_m0, timeout=300)
app = modal.App("assignment02-m0")


@app.function(
    image=image, gpu="B300", cpu=1, memory=2048, timeout=120,
    retries=0, max_containers=1, scaledown_window=2,
)
def run_m0():
    records = []

    def record(label, argv, timeout=10):
        item = run_command(argv, timeout=timeout)
        item["label"] = label
        records.append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)
        return item

    record("nvidia_smi", ["nvidia-smi"])
    record("nvcc_version", ["nvcc", "--version"])
    compute_apps = [
        "nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv",
    ]
    record("compute_apps_before", compute_apps)
    matched = record("run_100f", [str(REMOTE_ARTIFACTS / "first_mma_100f")], 20)
    mismatched = record("run_120a", [str(REMOTE_ARTIFACTS / "first_mma_120a")], 20)
    record("compute_apps_after", compute_apps)

    matched_pass = (matched["returncode"] == 0
                    and "PASS" in matched["stdout"].splitlines())
    artifacts = {
        "build.json": (REMOTE_ARTIFACTS / "build.json").read_bytes(),
        "first_mma_100f.ptx": (REMOTE_ARTIFACTS / "first_mma_100f.ptx").read_bytes(),
    }
    # These snapshots come from the image that ran, not from an assumed clean
    # checkout. They remain accurate if the local working tree has changed.
    for name in INPUT_FILES:
        artifacts[f"inputs/{name}"] = (REMOTE_CUDA / name).read_bytes()
    return {
        "completed_at_utc": utc_now(), "base_image": BASE_IMAGE,
        "remote_python": platform.python_version(),
        "remote_modal_version": distribution_version_or_none("modal"),
        "commands": records,
        "validation": {
            "matched_100f_pass": matched_pass,
            "mismatched_120a_returncode": mismatched["returncode"],
            "mismatched_120a_error": mismatched.get("error"),
        },
        "artifacts": artifacts,
    }


@app.local_entrypoint()
def main():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = HERE / "results" / f"m0-{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).read_bytes()
    (output / "modal_m0.py").write_bytes(script)
    metadata = {
        "started_at_utc": utc_now(), "local_python": platform.python_version(),
        "local_modal_version": importlib.metadata.version("modal"),
        "base_image": BASE_IMAGE, "runner_sha256": sha256(script),
        "git_head": run_command(["git", "rev-parse", "HEAD"], cwd=REPOSITORY),
        "git_status": run_command(
            ["git", "status", "--porcelain=v1"], cwd=REPOSITORY,
        ),
    }
    (output / "local-metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    try:
        result = run_m0.remote()
    except Exception as exc:
        (output / "run-error.json").write_text(
            json.dumps({"error_type": type(exc).__name__, "error": str(exc),
                        "recorded_at_utc": utc_now()}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Saved partial records: {output}")
        raise

    artifacts = result.pop("artifacts")
    result["saved_artifact_sha256"] = {}
    for name, data in artifacts.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        result["saved_artifact_sha256"][name] = sha256(data)
    (output / "run.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Saved: {output}")
    if not result["validation"]["matched_100f_pass"]:
        raise RuntimeError("100f binary did not report PASS; inspect run.json.")
    # The 120a result is experimental evidence; its nonzero exit is expected
    # and must not make the entire Modal invocation fail or trigger retries.
