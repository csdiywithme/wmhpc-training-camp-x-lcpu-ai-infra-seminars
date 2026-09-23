"""GPU acceptance CLI. Numerical validation only; timings are not benchmarks.

python validation/run_validation.py calibrate --tier quick --output calibration.json
python validation/run_validation.py verify --manifest calibration.json --output heldout.json
python validation/run_validation.py verify --manifest calibration.json \
    --adapter candidate_module:run --output candidate-heldout.json
"""

import argparse
import importlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import traceback

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validation import suite


def environment():
    import triton
    return {
        "python": platform.python_version(), "torch": torch.__version__,
        "triton": triton.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "scope": "vendored baseline with harness PDL disabled; eager execution",
    }


def hard_failures(metrics):
    p = suite.POLICY
    failures = []
    limits = {
        "active_nonfinite": 0, "max_abs": p["absolute_error_cap"],
        "row_nrmse_max": p["row_nrmse_cap"], "global_nrmse": p["global_nrmse_cap"],
        "elementwise_scaled_max": 1.0,
    }
    for key, limit in limits.items():
        if metrics[key] > limit:
            failures.append(f"{key}={metrics[key]:.6g} exceeds hard cap {limit:.6g}")
    return failures


def failures_for(metrics, bounds=None):
    failures = hard_failures(metrics)
    if bounds is not None:
        for name, limit in bounds.items():
            if metrics[name] > limit:
                failures.append(f"{name}={metrics[name]:.6g} exceeds frozen {limit:.6g}")
    return failures


def freeze_thresholds(records):
    if any(record["failures"] for record in records):
        raise ValueError("Baseline violated a predeclared cap; no acceptance manifest is frozen")
    observed = {}
    for record in records:
        if record["metrics"]["active_rows"] == 0:
            continue
        family = observed.setdefault(record["family"], {
            "max_abs": 0.0, "row_nrmse_max": 0.0, "global_nrmse": 0.0,
        })
        for name in family:
            family[name] = max(family[name], record["metrics"][name])
    p = suite.POLICY
    settings = {
        "max_abs": (p["absolute_error_floor"], p["absolute_error_cap"], p["absolute_error_margin"]),
        "row_nrmse_max": (p["row_nrmse_floor"], p["row_nrmse_cap"], 0.0),
        "global_nrmse": (p["global_nrmse_floor"], p["global_nrmse_cap"], 0.0),
    }
    thresholds = {}
    for family, metrics in observed.items():
        thresholds[family] = {
            name: min(cap, max(floor, p["baseline_multiplier"] * metrics[name] + margin))
            for name, (floor, cap, margin) in settings.items()
        }
    return thresholds, observed


@torch.inference_mode()
def run_matrix(specs, seeds, adapter, thresholds=None, label="baseline"):
    records = []
    for spec in specs:
        for seed in seeds:
            case = suite.make_case(spec, seed)
            reference = suite.gold(case)
            original_output = None
            cases = [case] + (suite.variants(case) if spec.metamorphic else [])
            for variant in cases:
                record = {"adapter": label, "case_id": spec.case_id, "seed": seed,
                          "variant": variant["variant"], "family": spec.family}
                try:
                    variant_reference = reference if variant is case else suite.gold(variant)
                    semantic_difference = (variant_reference - reference).abs().max().item()
                    if semantic_difference > 1e-10:
                        raise ValueError("Metamorphic builder changed the mathematical input")
                    output = adapter(variant)
                    torch.cuda.synchronize()
                    metrics = suite.evaluate(variant, output, variant_reference)
                    bounds = thresholds.get(spec.family) if thresholds is not None else None
                    if thresholds is not None and metrics["active_rows"] and bounds is None:
                        raise ValueError(f"No frozen calibration for family {spec.family}")
                    record.update(metrics=metrics, semantic_gold_max_abs=semantic_difference,
                                  failures=failures_for(metrics, bounds))
                    if original_output is None:
                        original_output = output.clone()
                    else:
                        invariant = suite.evaluate(variant, output, original_output.double())
                        record["invariance_metrics"] = invariant
                        # Triangle inequality allows twice each gold-error bound.
                        # Relative-to-near-zero original is not used here.
                        abs_bound = 2 * (bounds["max_abs"] if bounds else suite.POLICY["absolute_error_cap"])
                        row_bound = 2 * (bounds["row_nrmse_max"] if bounds else suite.POLICY["row_nrmse_cap"])
                        if invariant["active_nonfinite"] or invariant["max_abs"] > abs_bound or invariant["row_nrmse_max"] > row_bound:
                            record["failures"].append("Metamorphic output changed beyond the predeclared two-error allowance")
                except Exception as error:
                    record.update(failures=[f"{type(error).__name__}: {error}"],
                                  traceback=traceback.format_exc(limit=6), metrics=None)
                    records.append(record)
                    print(json.dumps(record), flush=True)
                    # A CUDA fault may poison the process; continuing can obscure
                    # the original failure. Return the partial record explicitly.
                    return records
                records.append(record)
                print(json.dumps({k: v for k, v in record.items() if k != "invariance_metrics"}), flush=True)
            del reference, original_output, cases, case
    return records


def load_adapter(target):
    if target is None:
        return suite.baseline
    module_name, attribute = target.rsplit(":", 1)
    if module_name.endswith(".py"):
        descriptor = importlib.util.spec_from_file_location("_c2_candidate_adapter", module_name)
        module = importlib.util.module_from_spec(descriptor)
        sys.modules[descriptor.name] = module
        descriptor.loader.exec_module(module)
    else:
        module = importlib.import_module(module_name)
    function = getattr(module, attribute)
    if not callable(function):
        raise TypeError("Adapter must be callable(case) -> output Tensor")
    return function


def record_count(specs, seeds):
    return sum((4 if spec.metamorphic and spec.decode_query_len == 1 else
                3 if spec.metamorphic else 1) * len(seeds) for spec in specs)


def successful(records, expected):
    return len(records) == expected and not any(record["failures"] for record in records)


def write_new(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(document, handle, indent=2, allow_nan=False)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("probe", "calibrate", "verify"))
    parser.add_argument("--tier", choices=("quick", "full"), default="quick")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--adapter", help="module:function or /absolute/file.py:function")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; manifests/results are immutable. Choose a new path.")
    if not torch.cuda.is_available():
        parser.error("CUDA GPU required; local AST/compile checks do not execute this suite")
    if args.mode != "verify" and args.adapter:
        parser.error("Calibration/probe use only the untouched vendored baseline")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.mode == "verify":
        if args.manifest is None:
            parser.error("verify requires --manifest")
        manifest = json.loads(args.manifest.read_text())
        if manifest.get("status") != "CALIBRATION_FROZEN":
            parser.error("Manifest is not a successful frozen baseline calibration")
        tier = manifest["protocol"]["tier"]
        current_protocol = suite.protocol(tier)
        if suite.protocol_digest(current_protocol) != manifest["protocol_digest"]:
            parser.error("Source/test/protocol differs from calibration; do not reuse its thresholds")
        specs = suite.build_specs(tier)
        seeds = suite.HELDOUT_SEEDS
        bounds = manifest["thresholds"]
        baseline_records = run_matrix(specs, seeds, suite.baseline, bounds, "baseline-heldout")
        expected = record_count(specs, seeds)
        baseline_ok = successful(baseline_records, expected)
        candidate_records = (run_matrix(specs, seeds, load_adapter(args.adapter), bounds, args.adapter)
                             if args.adapter and baseline_ok else [])
        candidate_ok = successful(candidate_records, expected) if args.adapter else baseline_ok
        status = ("BASELINE_HOLDOUT_FAILED" if not baseline_ok else
                  "PASS" if candidate_ok else "CANDIDATE_FAILED")
        result = {"status": status, "environment": environment(),
                  "manifest": str(args.manifest), "protocol_digest": manifest["protocol_digest"],
                  "candidate": args.adapter, "expected_records_per_adapter": expected,
                  "baseline_records": baseline_records, "candidate_records": candidate_records}
    else:
        protocol = suite.protocol(args.tier)
        specs = suite.build_specs(args.tier)
        seeds = suite.CALIBRATION_SEEDS if args.mode == "calibrate" else suite.CALIBRATION_SEEDS[:1]
        records = run_matrix(specs, seeds, suite.baseline)
        ok = successful(records, record_count(specs, seeds))
        result = {"status": "PROBE_PASS" if ok else "BASELINE_FAILED", "environment": environment(),
                  "protocol": protocol, "protocol_digest": suite.protocol_digest(protocol), "records": records}
        if args.mode == "calibrate" and ok:
            thresholds, observed = freeze_thresholds(records)
            result.update(status="CALIBRATION_FROZEN", thresholds=thresholds, observed_baseline_maxima=observed)
    write_new(args.output, result)
    print(json.dumps({"status": result["status"], "output": str(args.output)}), flush=True)
    return 0 if result["status"] in ("PASS", "PROBE_PASS", "CALIBRATION_FROZEN") else 1


if __name__ == "__main__":
    raise SystemExit(main())
