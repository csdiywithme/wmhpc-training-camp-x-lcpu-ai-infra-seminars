"""Recover logged round means without inventing missing per-call samples."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--console", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads((args.run / "request.json").read_text())
    text = args.console.read_text()
    result = {
        "status": "complete_from_console", "request": request,
        "source": {
            "type": "console", "path": str(args.console.resolve()),
            "sha256": hashlib.sha256(args.console.read_bytes()).hexdigest(),
            "missing": ["per_call_samples", "gpu_uuid", "telemetry", "post_timing_error_details", "two_ncu_reports"],
            "post_timing_pass_inferred_from_case_complete": True,
            "reason": "Modal heartbeat/return connection failed after execution. Retained FunctionCall lookup returned NotFound. No experiment was repeated.",
        },
        "environment": {"gpu": "NVIDIA B300 SXM6 AC", "sm_count": None,
                        "compiled_sm_count": request["sm_count"], "arch": request["arch"],
                        "gpu_name_source": "nvidia-smi row in console; UUID and telemetry unavailable"},
        "protocol": {"source": "sources/benchmark.py and request.json",
                     "rounds": request["rounds"], "repeat": request["repeat"],
                     "dtype": "FP16 input/output, FP32 accumulation",
                     "include_cublas": False, "raw_samples_available": False},
        "cases": {}, "ncu_collection_from_console": [],
    }
    for case in request["cases"]:
        result["cases"][case["name"]] = {
            "status": "pending", "shape": case["shape"], "validation": {},
            "measurements": {}, "round_order": [[] for _ in range(request["rounds"])],
            "post_timing_validation": {},
        }
    for line in text.splitlines():
        if line.startswith("VALIDATION "):
            _, name, variant, payload = line.split(" ", 3)
            result["cases"][name]["validation"].setdefault(variant, []).append(json.loads(payload))
        elif line.startswith("WARMUP "):
            _, name, variant, calibration, warmup, count = line.split()
            result["cases"][name]["measurements"][variant] = {
                "calibration_ms": float(calibration), "warmup_calls": int(warmup),
                "calls_per_round": int(count), "round_means_ms": [None] * request["rounds"],
                "source": "logged round means; individual event samples were not recovered",
            }
        elif line.startswith("ROUND "):
            _, name, index, variant, mean = line.split()
            data = result["cases"][name]
            data["measurements"][variant]["round_means_ms"][int(index)] = float(mean)
            data["round_order"][int(index)].append(variant)
        elif line.startswith("CASE_COMPLETE "):
            _, name, payload = line.split(" ", 2)
            case = result["cases"][name]
            case["logged_medians_ms"] = json.loads(payload)
            case["status"] = "complete"
        elif line.startswith("NCU_RESULT "):
            result["ncu_collection_from_console"].append(line)
    for requested in request["cases"]:
        case = result["cases"][requested["name"]]
        if case["status"] != "complete":
            raise ValueError("No CASE_COMPLETE: " + requested["name"])
        m, n, k = case["shape"]
        for variant in requested["variants"]:
            validation = case["validation"][variant]
            if {v["seed"] for v in validation if v["passed"]} != set(request["seeds"]):
                raise ValueError("Incomplete seed validation")
            item = case["measurements"][variant]
            means = item["round_means_ms"]
            if any(value is None for value in means):
                raise ValueError("Incomplete round means")
            median = statistics.median(means)
            if abs(median - case["logged_medians_ms"][variant]) > 1e-12:
                raise ValueError("Logged median mismatch")
            item.update(median_ms=median, tflops=2*m*n*k/median/1e9,
                        round_cv_percent=100*statistics.stdev(means)/statistics.mean(means),
                        min_round_ms=min(means), max_round_ms=max(means))
            case["post_timing_validation"][variant] = {
                "passed": True,
                "source": "CASE_COMPLETE reached after mandatory checks",
                "details": "Inferred from frozen sources/benchmark.py control flow; error details unavailable",
            }
    output = args.run / "artifacts" / "console_results.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print("Recovered", sum(len(c["measurements"]) for c in result["cases"].values()),
          "measurements from complete round means; per-call samples remain unavailable")

if __name__ == "__main__":
    main()
