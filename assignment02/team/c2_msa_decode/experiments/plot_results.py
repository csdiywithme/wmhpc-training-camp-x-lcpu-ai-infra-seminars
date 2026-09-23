"""Publication/export figures from saved GPU measurements; no estimated values."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
baseline = ROOT / "experiments/results/bench-b300-20260913T015939Z/measurements.json"
rows = json.loads(baseline.read_text())
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 160,
                     "svg.fonttype": "none"})
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
for axis, tp in zip(axes, (1, 4)):
    for dtype, color in (("bf16", "#1766a4"), ("fp8", "#c45c20")):
        data = sorted((r for r in rows if r["tp"] == tp and r["dtype"] == dtype), key=lambda r:r["batch"])
        x = [r["batch"] for r in data]
        y = [r["chain_graph"]["median_us"] for r in data]
        axis.plot(x, y, "o-", label=dtype.upper() + " KV", color=color)
        for a, b in zip(x, y):
            axis.annotate(f"{b:.2f}", (a, b), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)
    axis.set_title(f"TP{tp}: {64 // tp} Q heads / {4 // tp} KV heads")
    axis.set_xticks((1, 4, 8, 16));axis.set_xlabel("Batch size")
    axis.grid(axis="y", alpha=.2);axis.legend(frameon=False)
axes[0].set_ylabel("Decode chain median (microseconds)")
fig.suptitle("B300: pinned Triton MSA decode baseline", fontsize=13)
fig.text(.5,.005,"CUDA Graph, repeated fixed addresses, PDL off; FP8 KV dequantized to BF16 before dot",ha="center",fontsize=8)
fig.tight_layout(rect=(0,.05,1,.95))
for ext in ("svg", "png"):
    fig.savefig(OUT / f"baseline_latency.{ext}", bbox_inches="tight")
plt.close(fig)

copies=json.loads((ROOT/"results/paged-copy-20260913T015626Z/measurements.json").read_text())
fig, ax=plt.subplots(figsize=(7.4,4))
for heads, size, color, marker in ((1,1,"#1766a4","o"),(1,2,"#184b70","s"),(4,1,"#c45c20","o"),(4,2,"#863b14","s")):
    speed=[]
    for b in (1,4,8,16):
        values={r['path']:r['median_us'] for r in copies if (r['heads'],r['element_bytes'],r['batch'])==(heads,size,b)}
        speed.append(values['sm_vector']/values['tma'])
    ax.plot((1,4,8,16),speed,marker=marker,color=color,label=f"TP{4//heads}, {size} byte/element")
ax.axhline(1,color="#777",linewidth=.8,linestyle="--")
ax.set_xticks((1,4,8,16));ax.set_xlabel("Batch size")
ax.set_ylabel("Vector copy time / TMA copy time")
ax.set_title("B300: full-page copy after two SM-side index loads")
ax.grid(axis="y",alpha=.2);ax.legend(frameon=False,ncol=2)
fig.text(.5,.005,"Raw-byte global-to-shared-to-global microexperiment; this is not attention speedup",ha="center",fontsize=8)
fig.tight_layout(rect=(0,.05,1,1))
for ext in ("svg","png"):
    fig.savefig(OUT/f"paged_tma_copy.{ext}",bbox_inches="tight")
print(f"Saved figures to {OUT}")
