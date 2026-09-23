"""Create a standalone plot from completed full-domain benchmark records.

Run summarize_runs.py --write first. Requires matplotlib==3.10.3.
Bars are input ranges, not confidence intervals. No profiler times are plotted.
"""
import csv
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
ORDER = {
    "c1": ["fused", "direct-rowmajor", "direct", "direct-release", "direct-wide", "direct-wide-rowmajor"],
    "c2": ["original", "coalesced", "coalesced_pad17", "coalesced_release", "coalesced_wide", "original_wide"],
}
LABELS = {
    "fused": "v1 fused", "direct-rowmajor": "Direct epilogue / row-major",
    "direct": "Direct epilogue / transposed", "direct-release": "Transposed + release",
    "direct-wide": "Transposed + release + wide load",
    "direct-wide-rowmajor": "Row-major + release + wide load",
    "original": "v1 original feed", "coalesced": "Coalesced feed",
    "coalesced_pad17": "Coalesced + padding",
    "coalesced_release": "Coalesced + release", "coalesced_wide": "Coalesced + release + wide load",
    "original_wide": "Original feed + release + wide load",
}


def main():
    data = json.loads((ROOT / "RUN_INDEX.json").read_text())
    selected = {}
    for run in data:
        if run["mode"] != "bench" or run["status"] != "COMPLETE":
            continue
        args = run.get("build_args") or []
        variant = args[1] if args else "original"
        for artifact in run["artifacts"]:
            expected = 6 if run["track"] == "c1" else 32
            if artifact.get("cases") != expected or "rows" not in artifact:
                continue
            ratios = [row["speedup_vs_original"] for row in artifact["rows"]]
            if len(ratios) != expected or not all(math.isfinite(x) and x > 0 for x in ratios):
                raise ValueError(f"Invalid full benchmark: {run['run']}")
            selected[run["track"], variant] = {
                "track": run["track"], "variant": variant, "run": run["run"], "cases": expected,
                "geomean": math.exp(statistics.mean(math.log(x) for x in ratios)),
                "minimum": min(ratios), "maximum": max(ratios), "artifact": artifact["file"],
            }
    rows = [selected[t, v] for t, variants in ORDER.items() for v in variants if (t, v) in selected]
    with (ROOT / "benchmark_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.1), layout="constrained")
    for axis, track, color in zip(axes, ("c1", "c2"), ("#2563eb", "#0f766e")):
        group = [row for row in rows if row["track"] == track]
        for index, row in enumerate(group):
            axis.plot([row["minimum"], row["maximum"]], [index, index], color=color, lw=2.3, alpha=0.55)
            axis.plot(row["geomean"], index, "o", color=color, markersize=7)
            axis.text(row["maximum"] * 1.075, index, f"{row['geomean']:.3f}x", va="center", fontsize=9)
        axis.set_yticks(range(len(group)), [LABELS[row["variant"]] for row in group])
        axis.invert_yaxis()
        axis.set_xscale("log")
        axis.set_xlim(0.025, 1.18)
        axis.set_xticks([0.03, 0.05, 0.1, 0.2, 0.5, 1.0], ["0.03", "0.05", "0.1", "0.2", "0.5", "1.0"])
        axis.axvline(1.0, linestyle="--", color="#475569", linewidth=1.2)
        axis.grid(axis="x", color="#e2e8f0", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)
        axis.set_xlabel("Original latency / candidate latency (log scale)")
        axis.set_title("C1: full forward, 6 inputs" if track == "c1" else "C2: partial + merge, 32 inputs", loc="left", weight="bold", pad=18)
    fig.suptitle("B300 experiments: correctness passed; every measured new backend is slower", fontsize=14, weight="bold")
    fig.supxlabel("Dots: geometric means. Lines: range across inputs, NOT confidence intervals. Original = 1.0. Profiler times excluded.", fontsize=9)
    fig.savefig(ROOT / "performance_comparison.png", dpi=180)
    fig.savefig(ROOT / "performance_comparison.svg")
    plt.close(fig)
    print(f"Plotted {len(rows)} completed full-domain benchmarks")


if __name__ == "__main__":
    main()
