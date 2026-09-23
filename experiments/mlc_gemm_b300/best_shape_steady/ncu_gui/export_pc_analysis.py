"""Read an existing .ncu-rep with NVIDIA's Python report interface; no GPU use."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import ncu_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = ncu_report.load_report(args.report)
    action = report.range_by_idx(0).action_by_idx(0)
    selected = [
        name for name in action.metric_names()
        if name.startswith(("smsp__pcsamp_", "memory_"))
        or name in ("inst_executed", "thread_inst_executed", "thread_inst_executed_true",
                    "derived__memory_l1_wavefronts_shared_excessive",
                    "derived__memory_l1_conflicts_shared_nway")
    ]
    metrics, by_pc = {}, {}
    for name in selected:
        metric = action.metric_by_name(name)
        if not metric.num_instances() or not metric.has_correlation_ids():
            continue
        ids = metric.correlation_ids()
        metrics[name] = {"total": metric.value(), "num_instances": metric.num_instances()}
        for i in range(metric.num_instances()):
            pc = ids.as_uint64(i)
            by_pc.setdefault(pc, {})[name] = metric.value(i)
    known = [pc for pc in by_pc if action.sass_by_pc(pc)]
    start = min(known)
    while action.sass_by_pc(start - 16):
        start -= 16
    # Include instructions between sampled PCs as well as the unsampled prologue.
    for pc in range(start, max(by_pc) + 16, 16):
        sass = action.sass_by_pc(pc)
        if sass:
            by_pc.setdefault(pc, {})
    rows = []
    for pc, values in sorted(by_pc.items()):
        source = action.source_info(pc)
        rows.append({"pc": hex(pc), "offset_from_first_sass": hex(pc - start),
                     "sass": action.sass_by_pc(pc),
                     "source_file": source.file_name() if source else None,
                     "source_line": source.line() if source else None,
                     "metrics": values})
    output = {
        "report": str(args.report.resolve()),
        "sha256": hashlib.sha256(args.report.read_bytes()).hexdigest(),
        "kernel": action.name(), "first_available_sass_pc": hex(start),
        "source_files": dict(action.source_files()), "metrics": metrics,
        "instructions": rows,
    }
    (args.output / "pc_metrics.json").write_text(json.dumps(output, indent=2) + "\n")
    with (args.output / "pc_metrics.csv").open("w", newline="") as f:
        fields = ["pc", "offset_from_first_sass", "sass"] + list(metrics)
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**{key: row[key] for key in fields[:3]}, **row["metrics"]})
    with (args.output / "annotated.sass.txt").open("w") as f:
        f.write("# Existing report only; offsets relative to first available SASS instruction.\n")
        f.write("# No CUDA line information was embedded; see analysis for manual semantic mapping.\n")
        for row in rows:
            nonzero = {k: v for k, v in row["metrics"].items() if v and
                       (k.startswith("smsp__pcsamp_") and not k.endswith("_not_issued")
                        or k in ("memory_l1_wavefronts_shared", "memory_l1_wavefronts_shared_ideal",
                                 "derived__memory_l1_wavefronts_shared_excessive"))}
            offset = int(row['offset_from_first_sass'], 16)
            label = f"+0x{offset:x}" if offset >= 0 else f"-0x{-offset:x}"
            sass = row['sass'] or "[SASS unavailable in report]"
            f.write(f"{row['pc']} {label:>7}  {sass}\n")
            if nonzero:
                f.write("    # " + json.dumps(nonzero) + "\n")
    print(json.dumps({"kernel": action.name(), "instructions": len(rows),
                      "pc_metrics": len(metrics), "output": str(args.output)}))


if __name__ == "__main__":
    main()
