"""Export the final paired experiment, including PDL regressions."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
rows = json.loads((ROOT / "experiments/candidate_summary.json").read_text())["aggregate"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 160,
                     "svg.fonttype": "none"})
fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.1), sharey=True)
for ax, pdl in zip(axes, (False, True)):
    for offset, dtype, color in ((-.16, "bf16", "#1766a4"), (.16, "fp8", "#c45c20")):
        data = [next(r for r in rows if (r["pdl"], r["tp"], r["batch"], r["dtype"]) ==
                     (pdl, tp, batch, dtype)) for tp in (1, 4) for batch in (1, 4, 8, 16)]
        x = [i + offset for i in range(8)]
        y = [r["speedup"] for r in data]
        lo = [r["speedup"] - r["seed_min_speedup"] for r in data]
        hi = [r["seed_max_speedup"] - r["speedup"] for r in data]
        ax.errorbar(x, y, yerr=[lo, hi], fmt="o", markersize=6, capsize=3,
                    color=color, label=dtype.upper() + " KV")
    ax.axhspan(.65, 1, color="#f7dfdc", alpha=.65)
    ax.axhline(1, color="#75443e", linewidth=1, linestyle="--")
    ax.axvline(3.5, color="#777", linewidth=.7, alpha=.5)
    ax.set_xticks(range(8), ["1", "4", "8", "16"] * 2)
    ax.set_xlabel("Batch     |     left: TP1; right: TP4")
    ax.set_title("PDL on: conditional benefit" if pdl else "PDL off: all measured shapes improve")
    ax.set_ylim(.65, 1.45)
    ax.grid(axis="y", alpha=.18)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
axes[0].set_ylabel("Baseline chain time / candidate chain time")
fig.suptitle("B300: original partial + one-warp feature-tiled merge", fontsize=13)
fig.text(.5, .055, "Points: ratio of mean seed medians. Whiskers: range across seeds 101/307, not confidence intervals.",
         ha="center", fontsize=8)
fig.text(.5, .025, "Same-device paired CUDA Graphs, 64 calls/graph, 21 randomized repeats; actual graph outputs checked.",
         ha="center", fontsize=8)
fig.tight_layout(rect=(0, .10, 1, .95))
for ext in ("png", "svg"):
    fig.savefig(ROOT / "figures" / f"candidate_speedup.{ext}", bbox_inches="tight")
print("Saved final candidate PNG and SVG")
