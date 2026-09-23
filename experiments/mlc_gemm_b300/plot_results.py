"""Create exportable benchmark plots from the recorded measurements."""
import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    data = json.loads((args.run / "artifacts/results.json").read_text())
    if data["status"] != "complete":
        raise ValueError("Only completed measurements can be plotted")
    items = data["measurements"]
    ids = list(range(1, 10)) + [0]
    names = ["v1  Sync + MMA*", "v2  K-loop accumulation*", "v3  Spatial tiling",
             "v4  TMA", "v5  Software pipeline", "v6  Persistent",
             "v7  Warp specialization", "v8  Two-CTA cluster",
             "v9  Multi-consumer", "cuBLAS"]
    colors = ["#9aafbf"] * 2 + ["#5688a6"] * 5 + ["#279a91", "#087e76", "#de9352"]
    times = [items[str(v)]["median_ms"] for v in ids]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.spines.left": False, "axes.edgecolor": "#d6dce0",
                         "text.color": "#183042", "axes.labelcolor": "#183042"})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1.3, 1]})
    fig.patch.set_facecolor("#fafcfd")
    for a in (ax, bx):
        a.set_facecolor("#fafcfd")
        a.set_axisbelow(True)
        a.grid(axis="x", color="#e3e8ec", linewidth=0.6)
        a.tick_params(axis="y", length=0)
    y = np.arange(len(ids))
    ax.barh(y, times, color=colors, height=0.65)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(min(times) / 2, max(times) * 4)
    ax.set_xlabel("GPU stream latency, ms  •  log scale  •  lower is better")
    ax.set_title("All nine steps", loc="left", fontweight="bold", pad=15)
    for yi, ms in zip(y, times):
        ax.text(ms * 1.06, yi, f"{ms:.4f}" if ms < 1 else f"{ms:.2f}", va="center", fontsize=9)
    fast_ids = list(range(3, 10)) + [0]
    tflops = [items[str(v)]["tflops"] for v in fast_ids]
    fast_names = ["v" + str(v) if v else "cuBLAS" for v in fast_ids]
    fy = np.arange(len(fast_ids))
    bx.barh(fy, tflops, color=colors[2:], height=0.65)
    bx.set_yticks(fy, fast_names)
    bx.invert_yaxis()
    bx.set_xlim(0, max(tflops) * 1.2)
    bx.set_xlabel("TFLOP/s  •  higher is better")
    bx.set_title("Throughput after parallel tiling", loc="left", fontweight="bold", pad=15)
    for yi, t in zip(fy, tflops):
        bx.text(t + max(tflops) * .02, yi, f"{t:,.1f}", va="center", fontsize=9)
    fig.suptitle("GEMM on Modal B300", x=.025, ha="left", fontsize=21, fontweight="bold")
    fig.text(.025, .913, "4096 × 4096 × 4096  |  FP16 → FP32 accumulation → FP16  |  all outputs verified on 3 seeds", fontsize=11)
    fig.text(.025, .047, "* v1/v2 are full-matrix adaptations of the tutorial's single-tile examples. v1 includes register reduction.", fontsize=9)
    fig.text(.025, .020, "Median of five round means; CUDA events; 256 MiB cache flush outside each interval; default dynamic clocks.", fontsize=9)
    fig.tight_layout(rect=(0, .08, 1, .89), w_pad=3.0)
    fig.savefig(args.run / "performance.png", dpi=180, facecolor=fig.get_facecolor())
    fig.savefig(args.run / "performance.svg", facecolor=fig.get_facecolor())

if __name__ == "__main__":
    main()
