"""Pre-registered new inputs under the existing C2 frozen numeric gate.

No old suite constants/files are changed. This reuses the original validator's
run_matrix, counting and failure logic with explicit independent seeds/specs.
"""
import hashlib
import json
import os
from pathlib import Path


def run(args):
    from nextgen import adapter
    from validation import suite
    from validation import run_validation as validator
    registration_path = Path(__file__).with_name("EXTRA_HELDOUT.json")
    registration_bytes = registration_path.read_bytes()
    registration = json.loads(registration_bytes)
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("status") != "CALIBRATION_FROZEN" or manifest["protocol"]["tier"] != "full":
        raise ValueError("Extra heldout requires the original successful full manifest")
    if suite.protocol_digest(suite.protocol("full")) != manifest["protocol_digest"]:
        raise ValueError("Original protocol digest mismatch")
    if manifest["protocol_digest"] != registration["original_protocol_digest"]:
        raise ValueError("Registration does not identify this original numeric gate")
    seeds = tuple(registration["seeds"])
    if len(set(seeds)) != len(seeds) or set(seeds) & set(registration["excluded_development_seeds"]):
        raise ValueError("New seed registration overlaps known development inputs")
    base_specs = suite.build_specs("full")
    extra_specs = []
    for shape in registration["additional_shape_families"]:
        for heads in registration["kv_heads"]:
            for storage in registration["storages"]:
                extra_specs.append(suite.CaseSpec(
                    f"{shape['name']}_h{heads}_{storage}", heads,
                    tuple(shape["seq_lens"]), shape["decode_query_len"], storage=storage))
    specs = base_specs + extra_specs
    expected = validator.record_count(specs, seeds)
    if expected != registration["expected_records_per_adapter"]:
        raise ValueError("Registration/validator expected record count differs")
    if validator.record_count(base_specs, seeds) != registration["expected_existing_shape_new_seed_records"]:
        raise ValueError("Base-domain expected count differs")
    if validator.record_count(extra_specs, seeds) != registration["expected_additional_shape_records"]:
        raise ValueError("Additional-shape expected count differs")
    ext = adapter.extension()
    binary = Path(ext.__file__)
    frozen = {
        "registration": registration,
        "registration_sha256": hashlib.sha256(registration_bytes).hexdigest(),
        "original_manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "original_protocol_digest": manifest["protocol_digest"],
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "binary_variant_id": ext.variant_id() if hasattr(ext, "variant_id") else 0,
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in Path(__file__).parent.iterdir()
                          if p.suffix in (".py", ".cu", ".cpp", ".json")},
        "expected_records_per_adapter": expected,
        "scope": "New random inputs plus registered new shapes; existing numeric gate; eager, single GPU, PDL off",
    }
    # Persist candidate, dispatch, binary and input registration BEFORE any new
    # test input is generated. Root additionally archives the complete sources.
    validator.write_new(args.output / "extra-heldout-freeze.json", frozen)
    os.environ["C2_NEXTGEN_AUDIT"] = str((args.output / "extra-adapter-calls.jsonl").resolve())
    bounds = manifest["thresholds"]
    baseline_records = validator.run_matrix(specs, seeds, suite.baseline, bounds, "baseline-extra-heldout")
    baseline_ok = validator.successful(baseline_records, expected)
    candidate_records = (validator.run_matrix(specs, seeds, adapter.run, bounds, "tcgen05-extra-heldout")
                         if baseline_ok else [])
    candidate_ok = validator.successful(candidate_records, expected)
    status = "BASELINE_HOLDOUT_FAILED" if not baseline_ok else "PASS" if candidate_ok else "CANDIDATE_FAILED"
    result = {
        "status": status, "environment": validator.environment(), "freeze": frozen,
        "expected_records_per_adapter": expected,
        "existing_shapes_new_seeds_expected": validator.record_count(base_specs, seeds),
        "additional_shapes_expected": validator.record_count(extra_specs, seeds),
        "baseline_records": baseline_records, "candidate_records": candidate_records,
        "qualification": "Independent-input supplement under unchanged thresholds; does not replace seeds101/307 fixed regression",
    }
    validator.write_new(args.output / "extra-heldout.json", result)
    print(json.dumps({"status": status, "expected_records_per_adapter": expected,
                      "seeds": seeds, "output": str(args.output / "extra-heldout.json")}), flush=True)
    if status != "PASS":
        raise RuntimeError(f"Independent-input validation returned {status}; frozen thresholds were not changed")
