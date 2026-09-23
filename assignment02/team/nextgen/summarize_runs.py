"""Summarize preserved runs without importing CUDA, Torch or Modal.

Usage: python summarize_runs.py [--write]
--write refreshes derived RUN_INDEX.json; original run artifacts are never edited.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent


def read(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def geometric(values):
    return math.exp(statistics.mean(math.log(x) for x in values)) if values else None


def summarize(folder):
    request = read(folder / "request.json", {})
    state = read(folder / "state.json", {})
    result = {"run": folder.name, "track": request.get("track"),
              "mode": request.get("mode"), "build_args": request.get("build_args"),
              "extra_args": request.get("extra_args"),
              "status": state.get("status", "NO_STATE"),
              "app_id": state.get("app_id"), "artifacts": []}
    for path in sorted((folder / "artifacts").glob("*.json")):
        if path.name.startswith(("verify-", "long-verify-", "smoke-", "bench-")):
            obj = read(path)
            item = {"file": str(path.relative_to(HERE)), "status": obj.get("status"),
                    "cases": len(obj.get("rows", []))}
            if path.name.startswith("bench-"):
                item["rows"] = [{"case": row["case"],
                                  "median_us": {k: statistics.median(v) for k, v in row["timing"].items()},
                                  "speedup_vs_original": row["full_speedup"]} for row in obj["rows"]]
                item["geomean_speedup_vs_original"] = geometric([r["speedup_vs_original"] for r in item["rows"]])
            result["artifacts"].append(item)
        elif path.name == "paired.json":
            rows = read(path)
            result["artifacts"].append({"file": str(path.relative_to(HERE)), "cases": len(rows),
                "rows": [{k: row[k] for k in ("tp", "batch", "storage", "seed", "median_us", "speedup_vs_original", "speedup_vs_merge_only")} for row in rows],
                "geomean_speedup_vs_original": geometric([r["speedup_vs_original"] for r in rows]),
                "geomean_speedup_vs_merge_only": geometric([r["speedup_vs_merge_only"] for r in rows])})
        elif path.name == "progress.json":
            obj = read(path)
            result["artifacts"].append({"file": str(path.relative_to(HERE)),
                "status": obj.get("status"), "cases": len(obj.get("checks", {})),
                "scope": "Graph stage diagnosis; checks are not independent random cases"})
        elif path.name in ("smoke.json", "heldout.json", "extra-heldout.json"):
            obj = read(path)
            result["artifacts"].append({"file": str(path.relative_to(HERE)),
                "status": ("PASS" if all(r.get("status") == "PASS" for r in obj) else "FAIL") if isinstance(obj, list) else obj.get("status"),
                "cases": len(obj) if isinstance(obj, list) else obj.get("expected_records_per_adapter")})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    rows = [summarize(p) for p in sorted((HERE / "runs").iterdir()) if p.is_dir()]
    if args.write:
        (HERE / "RUN_INDEX.json").write_text(json.dumps(rows, indent=2) + "\n")
    for row in rows:
        summaries = []
        for item in row["artifacts"]:
            text = f"{Path(item['file']).name}: {item.get('status', '')} n={item.get('cases')}"
            if "geomean_speedup_vs_original" in item:
                text += f" speedup={item['geomean_speedup_vs_original']:.6f}"
            summaries.append(text)
        print(row["run"], row["status"], " | ".join(summaries))


if __name__ == "__main__":
    main()
