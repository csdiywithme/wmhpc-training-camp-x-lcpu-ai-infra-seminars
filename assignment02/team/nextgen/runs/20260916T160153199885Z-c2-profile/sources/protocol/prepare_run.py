"""Freeze only required experimental sources before a bounded Modal run."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
TEAM = HERE.parent
ROOT = TEAM.parent.parent
SUFFIXES = {".py", ".cu", ".cuh", ".cpp", ".cc", ".c", ".h", ".hpp", ".md", ".json"}
SKIP = {"results", "runs", "build", "__pycache__", ".git", ".venv"}


def source_files(directory: Path):
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory)
        if path.is_file() and path.suffix in SUFFIXES and not (set(relative.parts) & SKIP):
            yield path, relative


def prepare(track: str, mode: str, gpu: str, extra: list[str], variant: str | None = None, reuse_build: Path | None = None,
            qualification: Path | None = None, candidate_dir: Path | None = None) -> Path:
    if mode == "profile" and track != "c2":
        raise ValueError("Only C2 currently provides the bounded NVTX profile entrypoint")
    track_root = TEAM / ("c1_flashkda" if track == "c1" else "c2_msa_decode")
    candidate = candidate_dir.resolve() if candidate_dir is not None else track_root / "nextgen"
    if track == "c1" and variant is None:
        variant = "fused"
    for name in ("build.py", "run.py"):
        if not (candidate / name).is_file():
            raise FileNotFoundError(f"Candidate not ready: {candidate / name}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder = HERE / "runs" / f"{stamp}-{track}-{mode}"
    folder.mkdir(parents=True, exist_ok=False)
    sources = folder / "sources"
    records = []

    def save(source: Path, relative: Path):
        source = source.resolve()
        target = sources / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        data = source.read_bytes()
        target.write_bytes(data)
        records.append({"path": str(relative), "sha256": hashlib.sha256(data).hexdigest(),
                        "bytes": len(data), "original": str(source.relative_to(ROOT))})

    for path, relative in source_files(candidate):
        save(path, Path(track) / ("nextgen" if track == "c2" else "") / relative)
    if track == "c1":
        for name in ("run_experiments.py", "fla_kda_ref/naive.py"):
            save(track_root / name, Path("c1_reference") / name)
    else:
        for directory in ("harness", "validation"):
            for path, relative in source_files(track_root / directory):
                save(path, Path("c2") / directory / relative)
        for name in ("candidate.py", "vllm_msa_ref/sparse_attn.py", "experiments/baseline_workload.py", "experiments/candidate_workload.py"):
            save(track_root / name, Path("c2") / name)
        calibration = track_root / "results/candidate-calibrate-20260913T020759Z/calibration.json"
        save(calibration, Path("c2/validation/frozen_calibration.json"))
    for name in ("README.md", "JOURNAL.md", "HYPOTHESES.md", "ACCEPTANCE.md", "base_image.py", "modal_runner.py", "prepare_run.py"):
        path = HERE / name
        if path.is_file():
            save(path, Path("protocol") / name)
    if qualification is not None:
        if track != "c1":
            raise ValueError("qualification-json is currently used by the C1 benchmark entrypoint")
        save(qualification, Path("protocol/qualification.json"))
        extra = extra + ["--qualification-json", "/opt/nextgen/protocol/qualification.json"]
    request = {"track": track, "mode": mode, "gpu": gpu,
               "arch": "103a" if gpu == "B300" else "100a",
               "profile": "simidawhu", "created_at": datetime.now(timezone.utc).isoformat(),
               "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "extra_args": extra, "build_args": ["--variant", variant] if variant is not None else [],
               "status": "PREPARED"}
    if reuse_build is not None:
        old_request = json.loads((reuse_build / "request.json").read_text())
        old_build = json.loads((reuse_build / "build.json").read_text())
        if not old_build["success"] or any(old_request.get(k) != request.get(k) for k in ("track", "arch", "build_args")):
            raise ValueError("Cannot reuse a failed build or different track/arch/variant")
        def compiled_sources(rows):
            return {r["path"]: r["sha256"] for r in rows
                    if r["path"].startswith(track + "/")
                    and (Path(r["path"]).suffix in {".cu", ".cuh", ".cpp", ".h", ".hpp"}
                         or Path(r["path"]).name == "build.py")}
        previous = json.loads((reuse_build / "source_manifest.json").read_text())
        if compiled_sources(previous) != compiled_sources(records):
            raise ValueError("Compile sources changed; a fresh build is required")
        bundle = (reuse_build / "build.tar.gz").read_bytes()
        digest = hashlib.sha256(bundle).hexdigest()
        if digest != old_build["bundle_sha256"]:
            raise ValueError("Prior build bundle hash mismatch")
        (folder / "reused_build.tar.gz").write_bytes(bundle)
        (folder / "reused_build.json").write_text(json.dumps(old_build, indent=2) + "\n")
        request["reuse_build"] = {"original_run": str(reuse_build.resolve()), "bundle_sha256": digest}
    (folder / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    (folder / "source_manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    return folder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=("c1", "c2"), required=True)
    parser.add_argument("--mode", choices=("smoke", "verify", "bench", "profile"), default="smoke")
    parser.add_argument("--gpu", choices=("B200", "B300"), default="B300")
    parser.add_argument("--variant", choices=("fused", "split-qk", "split-final", "split-both", "baseline", "direct", "direct-rowmajor", "original", "coalesced", "coalesced_pad17"))
    parser.add_argument("--reuse-build-from", type=Path)
    parser.add_argument("--qualification-json", type=Path)
    parser.add_argument("--candidate-dir", type=Path, help="Freeze an earlier candidate source snapshot for a controlled comparison")
    parser.add_argument("extra", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    extra = args.extra[1:] if args.extra[:1] == ["--"] else args.extra
    print(prepare(args.track, args.mode, args.gpu, extra, args.variant, args.reuse_build_from, args.qualification_json, args.candidate_dir))


if __name__ == "__main__":
    main()
