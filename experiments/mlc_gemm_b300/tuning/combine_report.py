"""Combine two finished B300 runs without launching or repeating GPU work.

python combine_report.py --initial runs/20260918T092500Z \
    --refinement refinement/runs/<timestamp>

Incomplete benchmark results are rejected. Explicit console-derived results
may preserve complete round means without pretending per-call data survived.
Initial controls are historical, not contemporaneous refinement controls.
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics


PEAK = 2250.0
INITIAL_VARIANTS = ("original", "fenced", "wide", "full_late", "full_early", "cublas")
REFINEMENT_VARIANTS = ("release", "k128")
CONSOLE_POSTCHECK_SOURCE = "CASE_COMPLETE reached after mandatory checks"
METRICS = {
    "tensor_elapsed": "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "tensor_active": "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    "sm_active": "sm__cycles_active.avg.pct_of_peak_sustained_elapsed",
    "dram_pct": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram_bytes_per_second": "dram__bytes.sum.per_second",
    "l2_hit": "lts__t_sector_hit_rate.pct",
    "occupancy": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "physical_waves": "launch__waves_per_multiprocessor",
}


def number(value, digits=2):
    return "—" if value is None else f"{value:,.{digits}f}"


def numeric(value):
    try:
        result = float(str(value).replace(",", "").rstrip("%"))
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def ratio(numerator, denominator):
    return numerator / denominator if numerator is not None and denominator and denominator > 0 else None


def table(headers, rows):
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows),
    ])


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def link(path, output, label=None):
    relative = Path(os.path.relpath(path, output)).as_posix()
    return f"[{label or path.name}](<{relative}>)"


def load_run(path, role):
    path = path.resolve()
    source = path / "artifacts/results.json"
    if not source.exists() and role == "refinement":
        source = path / "artifacts/console_results.json"
    result = json.loads(source.read_text())
    source_info = result.get("source", {})
    console_derived = result.get("status") == "complete_from_console"
    if console_derived:
        if (role != "refinement" or source_info.get("type") != "console" or
                "per_call_samples" not in source_info.get("missing", []) or
                source_info.get("post_timing_pass_inferred_from_case_complete") is not True):
            raise ValueError(f"{role}: console recovery lacks explicit provenance/evidence: {source}")
    elif result.get("status") != "complete":
        raise ValueError(f"{role}: refusing incomplete benchmark {source}: {result.get('status')}")
    request = result["request"]
    rows = []
    workers = request["sm_count"] // 2
    for requested_case in request["cases"]:
        name = requested_case["name"]
        case = result["cases"][name]
        accepted_statuses = ("complete", "complete_from_console") if console_derived else ("complete",)
        if case.get("status") not in accepted_statuses or case["shape"] != requested_case["shape"]:
            raise ValueError(f"{role}/{name}: incomplete case or mismatched shape")
        m, n, k = case["shape"]
        tasks = (m // 512) * (n // 256)
        variants = list(requested_case["variants"])
        if request.get("include_cublas", True):
            variants.append("cublas")
        for variant in variants:
            item = case["measurements"].get(variant, {})
            means = item.get("round_means_ms", [])
            samples = item.get("samples_ms", [])
            seeds = case.get("validation", {}).get(variant, [])
            passed = {record.get("seed") for record in seeds if record.get("passed") is True}
            post = case.get("post_timing_validation", {}).get(variant, {})
            if (len(means) != request["rounds"] or not means or
                    any(not math.isfinite(x) or x <= 0 for x in means) or
                    not all(seed in passed for seed in request["seeds"]) or
                    any(record.get("passed") is not True for record in seeds) or
                    post.get("passed") is not True):
                raise ValueError(f"{role}/{name}/{variant}: incomplete rounds or correctness checks")
            if console_derived:
                if "samples_ms" in item:
                    raise ValueError(f"{role}/{name}/{variant}: console recovery must not fabricate per-call samples")
                if post.get("source") != CONSOLE_POSTCHECK_SOURCE:
                    raise ValueError(f"{role}/{name}/{variant}: missing CASE_COMPLETE postcheck evidence")
            elif len(samples) != len(means) or any(len(batch) != item.get("calls_per_round") for batch in samples):
                raise ValueError(f"{role}/{name}/{variant}: incomplete raw timing samples")
            elapsed = statistics.median(means)
            if item.get("median_ms") is not None and not math.isclose(elapsed, item["median_ms"], rel_tol=1e-9):
                raise ValueError(f"{role}/{name}/{variant}: saved median disagrees with raw round means")
            throughput = 2 * m * n * k / elapsed / 1e9
            rows.append({
                "run_role": role, "run_id": path.name, "case": name, "variant": variant,
                "M": m, "N": n, "K": k, "cluster_tasks": tasks,
                "logical_waves": math.ceil(tasks / workers),
                "model_slot_utilization_pct": 100 * tasks / (workers * math.ceil(tasks / workers)),
                "median_ms": elapsed, "tflops": throughput, "nominal_peak_pct": throughput / PEAK * 100,
                "min_round_ms": min(means), "max_round_ms": max(means),
                "round_cv_pct": (100 * statistics.stdev(means) / statistics.mean(means)
                                 if len(means) > 1 else 0),
                "rounds": len(means), "calls_per_round": item["calls_per_round"],
                "seed_checks_passed": len(passed), "seed_checks_expected": len(request["seeds"]),
                "postcheck_passed": post["passed"], "postcheck_mismatches": post.get("mismatch_count"),
                "postcheck_evidence": post.get("source", "saved per-variant post-timing record"),
                "seeds": json.dumps(request["seeds"]),
                "source_type": "console_recovery" if console_derived else "raw_results_json",
                "per_call_samples_available": not console_derived,
                "reference_scope": "same initial run" if role == "initial" else "historical initial run",
                "initial_original_throughput_ratio": None,
                "initial_fenced_throughput_ratio": None,
                "initial_cublas_throughput_ratio": None,
                "source_results": str(source), "source_results_sha256": sha256(source),
            })
    return {"role": role, "path": path, "source": source, "result": result,
            "request": request, "rows": rows, "console_derived": console_derived,
            "index": {(row["case"], row["variant"]): row for row in rows}}


def add_historical_ratios(initial, refinement):
    for row in initial["rows"] + refinement["rows"]:
        for variant in ("original", "fenced", "cublas"):
            baseline = initial["index"].get((row["case"], variant))
            if baseline:
                if any(row[key] != baseline[key] for key in ("M", "N", "K")):
                    raise ValueError("Same case name has different shapes across runs: " + row["case"])
                row["initial_" + variant + "_throughput_ratio"] = baseline["median_ms"] / row["median_ms"]


def read_ncu(run):
    """Honor recovered initial data and successful refinement long-table rows."""
    path = run["path"]
    recovered_path = path / "ncu_metrics.json"
    summary_path = path / "artifacts/ncu/ncu_summary.json"
    records = []
    if recovered_path.exists():
        recovered = json.loads(recovered_path.read_text())
        for source in recovered.get("records", []):
            if source.get("collection_status") != "success":
                continue
            raw = source.get("raw_values", {})
            records.append({
                "run_role": run["role"], "case": source["case"], "variant": source["variant"],
                "values": {key: numeric(raw.get(metric)) for key, metric in METRICS.items()},
                "report": path / source["report_source"],
                "csv": path / source["csv_source"], "source": recovered_path,
            })
        return records, {"status": recovered.get("status"), "source": recovered_path,
                         "reason": recovered.get("reason", "")}
    if not summary_path.exists():
        if run["console_derived"]:
            return [], {"status": "not_retrieved", "source": summary_path,
                        "reason": "新增两份 profile 在控制台记录 complete，但远端报告、CSV 和指标值未能取回"}
        return [], {"status": "not_collected", "source": summary_path,
                    "reason": "没有保存 Nsight Compute summary"}
    summary = json.loads(summary_path.read_text())
    for source in summary.get("profiles", []):
        if source.get("status") != "complete":
            continue
        raw = {item["Metric Name"]: item.get("Metric Value")
               for item in source.get("metrics", []) if "Metric Name" in item}
        if not raw:
            continue
        stem = source["case"] + "__" + source["variant"]
        records.append({
            "run_role": run["role"], "case": source["case"], "variant": source["variant"],
            "values": {key: numeric(raw.get(metric)) for key, metric in METRICS.items()},
            "report": summary_path.parent / source.get("report", stem + ".ncu-rep"),
            "csv": summary_path.parent / (stem + ".csv"), "source": summary_path,
        })
    return records, {"status": summary.get("status"), "source": summary_path,
                     "reason": summary.get("reason", summary.get("error", ""))}


def plot(initial, refinement, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 1, figsize=(14, 10.5), constrained_layout=True,
                             gridspec_kw={"height_ratios": [1, 1.25]})
    ax = axes[0]
    sweep = [name for name in ("rect128", "rect144", "aligned148", "tail152")
             if (name, "original") in initial["index"]]
    x = np.arange(len(sweep))
    colors = {"original": "#51627a", "fenced": "#8873a6", "cublas": "#bf644b",
              "release": "#228d83", "k128": "#3678bd"}
    for variant in ("original", "fenced", "cublas"):
        rows = [initial["index"][(name, variant)] for name in sweep]
        values = [row["median_ms"] * 1000 for row in rows]
        ax.plot(x, values, "o-", label=variant + " (initial run)", color=colors[variant], linewidth=1.7)
        ax.fill_between(x, [row["min_round_ms"] * 1000 for row in rows],
                        [row["max_round_ms"] * 1000 for row in rows], color=colors[variant], alpha=0.12)
    ax.set_xticks(x, [f"{initial['index'][(name, 'original')]['cluster_tasks']} tasks\n"
                     f"{initial['index'][(name, 'original')]['logical_waves']} logical waves" for name in sweep])
    ax.set_ylabel("GEMM latency (microseconds)")
    ax.set_title("A  |  Wave boundary in one session: M=2048, K=4096, changing N", loc="left", weight="bold")
    ax.text(0.01, 0.97, "148 tasks = 74 + 74; 152 tasks = 74 + 74 + 4",
            transform=ax.transAxes, va="top", color="#555555", fontsize=10)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, 0.9), frameon=False)
    ax.set_ylim(bottom=0)

    ax = axes[1]
    names = [case["name"] for case in refinement["request"]["cases"]]
    x = np.arange(len(names))
    variants = ("original", "fenced", "cublas", "release", "k128")
    width = 0.145
    for column, variant in enumerate(variants):
        historical = variant not in REFINEMENT_VARIANTS
        run = initial if historical else refinement
        rows = [run["index"].get((name, variant)) for name in names]
        values, lower, upper = [], [], []
        for row in rows:
            if row is None:
                values.append(float("nan")); lower.append(0); upper.append(0)
                continue
            values.append(row["tflops"])
            flop_factor = 2 * row["M"] * row["N"] * row["K"] / 1e9
            lower.append(max(0, row["tflops"] - flop_factor / row["max_round_ms"]))
            upper.append(max(0, flop_factor / row["min_round_ms"] - row["tflops"]))
        ax.bar(x + (column - 2) * width, values, width,
               label=variant + (" (historical)" if historical else " (new session)"),
               color=colors[variant], hatch="//" if historical else None,
               edgecolor="white", linewidth=0.7, yerr=[lower, upper], capsize=2,
               error_kw={"elinewidth": 0.75})
    ax.axhline(PEAK, color="#555555", linestyle="--", linewidth=1,
               label="Nominal dense FP16 peak: 2250 TFLOPS")
    labels = []
    for name in names:
        row = refinement["index"][(name, refinement["request"]["cases"][names.index(name)]["variants"][0])]
        labels.append(f"{name}\n{row['M']} x {row['N']} x {row['K']}")
    ax.set_xticks(x, labels, fontsize=9)
    ax.set_ylabel("Throughput (TFLOPS)")
    ax.set_title("B  |  Pipeline experiments with reused historical controls", loc="left", weight="bold")
    ax.set_ylim(0, max(PEAK * 1.25, max(row["tflops"] for row in initial["rows"] + refinement["rows"]) * 1.25))
    ax.legend(ncol=3, fontsize=9, loc="upper left", frameon=False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.2)
    fig.suptitle("B300: aligned workloads and pipeline variants", fontsize=18, weight="bold")
    refinement_source = "Refinement uses saved console round means; per-call samples/telemetry were not retrieved.\n" if refinement["console_derived"] else ""
    fig.supxlabel(f"Initial: {initial['path'].name}  |  Refinement: {refinement['path'].name}\n"
                  "Bars/lines: median of round means; whiskers/bands: observed round range, not confidence intervals.\n"
                  + refinement_source +
                  "Historical and new bars come from different GPU sessions; small differences are not causal evidence.",
                  fontsize=9, color="#555555")
    fig.savefig(output / "final_comparison.png", dpi=180)
    fig.savefig(output / "final_comparison.svg")
    plt.close(fig)


def environment_row(run):
    if run["console_derived"]:
        return [run["role"], run["path"].name, "B300（请求；GPU metadata 未取回）",
                "未取回", "未取回", "未取回", "未取回"]
    env = run["result"].get("environment", {})
    text = env.get("initial_telemetry", {}).get("stdout", "")
    fields = next(csv.reader([text.strip()]), [])
    return [run["role"], run["path"].name, env.get("gpu", "未记录"),
            env.get("sm_count", "未记录"), fields[1].strip() if len(fields) > 1 else "未记录",
            fields[2].strip() if len(fields) > 2 else "未记录",
            fields[7].strip() if len(fields) > 7 else "未记录"]


def generate_report(initial, refinement, output, ncu_records, ncu_status):
    rows = initial["rows"] + refinement["rows"]
    custom = [row for row in rows if row["variant"] != "cublas"]
    winner = max(custom, key=lambda row: row["tflops"])
    overall = max(rows, key=lambda row: row["tflops"])
    new_winner = max(refinement["rows"], key=lambda row: row["tflops"])
    old_square = initial["index"].get(("square128", "original"))
    lines = ["# B300：148 任务对齐与流水线优化实测", "",
             f"全部已完成记录中，自定义 kernel 的最高吞吐来自 **{winner['case']} / {winner['variant']}"
             f"（{winner['run_role']}）**：**{number(winner['tflops'])} TFLOPS**、"
             f"{number(winner['median_ms'], 5)} ms，达到 2250 TFLOPS 标称稠密 FP16 峰值的 "
             f"**{number(winner['nominal_peak_pct'])}%**。"
             "最高值从所有自定义版本中选择，包含 original 和 fenced。", "",
             f"包含库参考在内的最高记录是 **{overall['case']} / {overall['variant']}**："
             f"{number(overall['tflops'])} TFLOPS。新增候选中最高记录为 "
             f"**{new_winner['case']} / {new_winner['variant']}**：{number(new_winner['tflops'])} TFLOPS。", "",
             "**新增 release/k128 与原版、fenced、cuBLAS 来自不同 B300 会话。** 按要求复用初次对照，"
             "没有重复测量历史版本。后文“历史比值”仅表示已保存时延的比值，"
             "不是同卡同批配对因果实验；几个百分点的差异不据此宣称显著改善。"
             "同一新会话内 release 与 k128 的比较条件更接近，但二者同时改变不同机制，不能单独归因。", ""]
    if refinement["console_derived"]:
        lines[2:2] = [
            "**数据来源说明：初次 54 组测量保留逐次原始样本和 6 份 NCU 报告；"
            "新增 8 组由已落盘控制台恢复完整 7 轮均值及三种子验证日志。** "
            "新会话云端工作已完成，但 Modal heartbeat 断线导致返回取消；"
            "逐次计时样本、GPU UUID/telemetry、计时后误差详情及两份新增 NCU 报告未取回。"
            "计时后验证通过由执行到 `CASE_COMPLETE` 的控制流证据推断，"
            "不是恢复了未取得的详细记录。收到不要重复的要求后，未重跑已有 GPU 实验。", "",
        ]
    lines[2:2] = [
        "**任务数对齐有明确收益；本次流水线改动尚未证明有稳定收益。** "
        "最高记录 wide 相对同形状 fenced 仅高约 0.52%，对应轮间 CV 为 2.80% 与 1.07%，"
        "不能把这个微小差距解释为确定的优化效果。", "",
    ]
    if old_square:
        lines += [f"相较初次 4096³ original 的 {number(old_square['tflops'])} TFLOPS，"
                  f"上述最高自定义记录吞吐为 {number(winner['tflops'] / old_square['tflops'], 3)}×。"
                  "此比较改变了矩阵形状或版本，表示工作负载吞吐提升，不表示同一 GEMM 的等量工作加速。", ""]
    lines += ["![形状与流水线对照](final_comparison.png)", "",
              "图中斜线柱是初次历史对照，实心柱是新会话。误差线和色带表示各轮均值的最小至最大范围，"
              "不是置信区间。两次运行均以 CUDA events 的轮均值中位数计算性能，"
              "`TFLOPS = 2MNK / (ms × 10^9)`。", "",
              "## 设备、测量与验证", "",
              table(["会话", "run", "GPU", "实际 SM", "GPU UUID", "driver", "开始时 SM clock"],
                    [environment_row(initial), environment_row(refinement)]), "",
              "已保存的开始时钟仅为一次 telemetry 快照，不代表整个测量过程锁频。代码配置使用默认动态频率/功率；"
              "逐次清理 256 MiB 缓冲区，清理在 CUDA events 之外；随机打乱版本顺序；不用 CUDA Graph。"
              "分配、编译、生成输入和校验均不计入时延，事件区间可能包含 host 提交间隙。", "",
              "FP16 输入/输出、FP32 累加；FP32 reference 禁用 TF32 并最终舍入为 FP16，"
              "逐元素 `atol=0.01, rtol=0.02`。每次初始验证前填 NaN 检查漏写；"
              "代码在全部计时之后另检查最终输出。", ""]
    validation_rows = []
    for run in (initial, refinement):
        rowset = run["rows"]
        validation_rows.append([run["role"], str(run["request"]["seeds"]),
                                f"{sum(row['seed_checks_passed'] for row in rowset)}/"
                                f"{sum(row['seed_checks_expected'] for row in rowset)}",
                                f"{sum(row['postcheck_passed'] for row in rowset)}/{len(rowset)}"
                                + ("（CASE_COMPLETE 推断）" if run["console_derived"] else "（明细已保存）"),
                                run["request"]["rounds"], run["request"]["repeat"]])
    lines += [table(["会话", "seeds", "种子验证通过", "计时后验证通过", "轮数", "每轮调用上限"], validation_rows), "",
              "round CV 是各轮均值的样本标准差除以均值。它反映本次运行的波动，"
              "不包含不同 GPU 会话的系统性偏移，也不代替独立重复实验。"
              "峰值分母统一为 2250 TFLOPS；它不是本卡当前时钟下实测的持续算力。"
              "规格口径见 [NVIDIA HGX](https://www.nvidia.com/en-us/data-center/hgx/)。", "",
              "## 正好 148 个逻辑任务", "",
              "kernel 固定启动 148 个 CTA，每两个 CTA 组成一个 cluster，共 74 个 cluster worker。"
              "每个逻辑任务计算一个 `512×256` 输出 tile。**148 指逻辑任务数，不是额外增加 grid CTA 数。**"
              "等时任务模型的槽位利用率为 `tasks / (74 × ceil(tasks/74))`，属于调度模型而非硬件计数器。", ""]
    if refinement["console_derived"]:
        lines += ["新会话缺少已取回的 GPU UUID 与 telemetry，无法核对两次物理卡身份、"
                  "时钟或温度差异；因此不能把微小历史比值变化归因于代码优化。"
                  "新会话的 round CV 和误差线仅由完整日志中的 7 个轮均值计算，未构造逐次计时样本。", ""]
    shape_rows = []
    for case in initial["request"]["cases"]:
        row = initial["index"][(case["name"], "original")]
        shape_rows.append([row["case"], f"{row['M']}×{row['N']}×{row['K']}", row["cluster_tasks"],
                           row["logical_waves"], number(row["model_slot_utilization_pct"]) + "%",
                           number(row["median_ms"], 5), number(row["tflops"])])
    lines += [table(["case", "M×N×K", "任务", "逻辑轮数", "模型槽位利用率", "original ms", "original TFLOPS"], shape_rows), ""]
    aligned = initial["index"].get(("aligned148", "original"))
    tail = initial["index"].get(("tail152", "original"))
    if aligned and tail:
        flop_growth = (tail["M"] * tail["N"] * tail["K"] / (aligned["M"] * aligned["N"] * aligned["K"]) - 1) * 100
        latency_growth = (tail["median_ms"] / aligned["median_ms"] - 1) * 100
        lines += [f"148→152 时 M、K 不变，算术工作量增加 **{number(flop_growth)}%**，"
                  f"benchmark 时延增加 **{number(latency_growth)}%**，吞吐从 {number(aligned['tflops'])} "
                  f"降至 {number(tail['tflops'])} TFLOPS。148 个任务为 74+74；152 个任务为 74+74+4。"
                  "下方 NCU 计数器提供了尾轮效应的独立硬件证据。", ""]
    lines += ["矩阵长宽比与缓存复用也会变化。square128→aligned148 不能把全部差异归给尾轮；"
              "aligned592 同时增大 M、K 与迭代数，也不能单独判定是哪一个因素带来收益。", "",
              "## 流水线实现与首次结果", "",
              table(["版本", "K tile / 输入 stage", "每 consumer 写回", "与 fenced 的主要差别", "SMEM/CTA"], [
                  ["original", "64 / 4", "4×64 列", "原教程版本，缺少补充的 TMEM fence", "225 KiB"],
                  ["fenced", "64 / 4", "4×64 列", "同步控制组", "225 KiB"],
                  ["wide", "64 / 3", "2×128 列", "同时扩大写回与减少输入 stage", "209 KiB"],
                  ["full_late", "64 / 2", "1×256 列", "整块写回后释放 accumulator", "225 KiB"],
                  ["full_early", "64 / 2", "1×256 列", "整块 TMEM 读出后提前释放 accumulator", "225 KiB"],
                  ["release", "64 / 4", "4×64 列", "最后一个 chunk 的 TMEM 读完成即释放", "225 KiB"],
                  ["k128", "128 / 2", "4×64 列", "加倍输入 K tile，输入容量保持不变", "225 KiB"],
              ]), "",
              "所有自定义版本保留两个 MMA consumer、双 CTA cluster、384 threads/CTA 和 512 列 TMEM。"
              "原始输出路径已使用 `cp.async.bulk.wait_group.read 0`，等待 TMA 不再读取源共享内存，"
              "不能说每个 chunk 都等待显存写入彻底完成。"
              "[PTX read-wait 语义](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-wait-group)", ""]
    old_rows = []
    for case in initial["request"]["cases"]:
        name = case["name"]
        old_rows.append([name] + [number(initial["index"][(name, variant)]["tflops"])
                                  for variant in INITIAL_VARIANTS])
    lines += [table(["case", *[variant + " TFLOPS" for variant in INITIAL_VARIANTS]], old_rows), ""]
    full_ratios = [row["initial_fenced_throughput_ratio"] for row in initial["rows"]
                   if row["variant"] in ("full_late", "full_early")]
    if full_ratios:
        lines += [f"full_late/full_early 的同会话吞吐/fenced 比值范围为 "
                  f"**{number(min(full_ratios), 3)}–{number(max(full_ratios), 3)}×**。"
                  "把输出合并为一次 TMA store 未保证净收益：输出缓冲扩大迫使输入 stage 从 4 降到 2，"
                  "可能削弱隐藏输入延迟的能力，也可能改变寄存器压力和同步等待。"
                  "这是对负结果的机制假设；未采集这些 full2stage 版本的 stall 归因，不能断言唯一原因。", ""]
    lines += ["## 新增候选：全部四种形状", "",
              "下列三个比值的分子均是初次 run 保存的同形状对照时延，分母是新候选时延；"
              "也等于新候选 TFLOPS 除以历史对照 TFLOPS。大于 1 表示记录数值更高。"
              "**历史比值不具有同卡同批因果解释，尤其不把几个百分点的差距称为显著加速。**", ""]
    for variant in REFINEMENT_VARIANTS:
        candidate_rows = []
        for case in refinement["request"]["cases"]:
            row = refinement["index"].get((case["name"], variant))
            if row:
                candidate_rows.append([row["case"], number(row["median_ms"], 5), number(row["tflops"]),
                                       number(row["initial_original_throughput_ratio"], 3),
                                       number(row["initial_fenced_throughput_ratio"], 3),
                                       number(row["initial_cublas_throughput_ratio"], 3),
                                       number(row["round_cv_pct"]),
                                       f"{row['seed_checks_passed']}/{row['seed_checks_expected']}",
                                       ("CASE_COMPLETE 推断通过" if row["source_type"] == "console_recovery"
                                        else "通过") if row["postcheck_passed"] else "失败"])
        lines += [f"### {variant}", "", table(["case", "ms", "TFLOPS", "相对历史 original ×",
                    "相对历史 fenced ×", "相对历史 cuBLAS ×", "round CV %", "种子验证", "计时后"], candidate_rows), ""]
    paired = []
    for case in refinement["request"]["cases"]:
        name = case["name"]
        release = refinement["index"].get((name, "release"))
        k128 = refinement["index"].get((name, "k128"))
        if release and k128:
            paired.append([name, number(k128["tflops"] / release["tflops"], 3),
                           number(release["round_cv_pct"]), number(k128["round_cv_pct"])])
    lines += ["两种新候选在同一新会话内的比较：", "",
              table(["case", "k128/release 吞吐比", "release CV %", "k128 CV %"], paired), "",
              "release 尝试增加最后一段写回与下一块 MMA 的重叠，保留四级输入流水线。"
              "k128 把每级 K 从 64 增至 128、深度从 4 降至 2，输入缓冲总容量不变；"
              "每次 `gemm_async` 展开 8 条 K16 MMA，总 MMA 算术量未减少，改变的是每级加载/提交次数。"
              "更少的控制操作与更粗的等待粒度同时存在，性能方向以实测为准。", "",
              "## Nsight Compute 机制证据", "",
              "初次 run 原 summary 的 failed 来自 wide CSV 解析器误判；已由逐份成功退出码、"
              "有效报告及原始 CSV 恢复硬件指标，原文件保持不变。详见 "
              + link(initial["path"] / "NCU_ANALYSIS.md", output, "初次 NCU 独立审查") + "。", "",
              "NCU 使用 kernel replay、cache-control=all、clock-control=none，一次目标 launch；"
              "计数器采集会改变执行条件，NCU 单次耗时不替代上面的 benchmark 排名。"
              "Tensor/elapsed、Tensor/active 是不同周期分母的管线活动指标，"
              "**不等于按 2MNK 计算的 2250 TFLOPS 峰值百分比**。"
              "[NVIDIA 指标定义](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-structure)", ""]
    ncu_rows = []
    for record in ncu_records:
        values = record["values"]
        ncu_rows.append([record["run_role"], record["case"] + "/" + record["variant"],
                         number(values["sm_active"]), number(values["tensor_elapsed"]),
                         number(values["tensor_active"]), number(values["dram_pct"]),
                         number(values["l2_hit"]), number(values["occupancy"])])
    if ncu_rows:
        lines += [table(["会话", "case/variant", "SM active %", "Tensor/elapsed %", "Tensor/active %",
                         "DRAM %", "L2 hit %", "warp occupancy %"], ncu_rows), ""]
    for role, status in ncu_status.items():
        count = sum(record["run_role"] == role for record in ncu_records)
        lines += [f"{role} NCU：状态 `{status['status']}`，本报告读取 {count} 个成功 profile。"
                  + (f"说明：{status['reason']}" if not count and status.get("reason") else ""), ""]
    if not any(record["run_role"] == "refinement" for record in ncu_records):
        lines += ["**实际可审查的 NCU 数据只有初次保存的六份。** 新增两份仅在日志中显示 complete，"
                  "对应报告文件和指标值未能取回，不列入下表或证据计数，也不据此给 release/k128 做硬件归因。"
                  if refinement["console_derived"] else
                  "新候选没有可用的已完成 NCU 指标，不能给 release/k128 编造硬件层面的归因。", ""]
    ncu_index = {(record["run_role"], record["case"], record["variant"]): record for record in ncu_records}
    a = ncu_index.get(("initial", "aligned148", "original"))
    b = ncu_index.get(("initial", "tail152", "original"))
    if a and b:
        av, bv = a["values"], b["values"]
        lines += [f"148→152 的 NCU SM active 从 **{number(av['sm_active'])}%→{number(bv['sm_active'])}%**，"
                  f"Tensor/elapsed 从 **{number(av['tensor_elapsed'])}%→{number(bv['tensor_elapsed'])}%**，"
                  f"而 Tensor/active 为 {number(av['tensor_active'])}% 与 {number(bv['tensor_active'])}%。"
                  "这支持第三轮少量任务拖长 kernel、更多 SM 提前空闲的解释。", ""]
    dram = [record["values"]["dram_pct"] for record in ncu_records if record["values"]["dram_pct"] is not None]
    if dram:
        lines += [f"成功 profile 的 DRAM 整体吞吐利用率范围为 {number(min(dram))}–{number(max(dram))}%。"
                  "这些平均指标不能直接定位 TMA、共享内存或 barrier 的瞬时等待。", ""]
    lines += ["`launch__waves_per_multiprocessor` 描述物理 grid CTA waves，不能替代 persistent kernel 内部逻辑任务轮数。"
              "低 warp occupancy 也不等于同百分比的 Tensor 算力；异步 MMA 和 warp specialization 的利用率需单独观察。", ""]
    reports = [record for record in ncu_records if record["case"] == "aligned148"]
    if reports:
        lines += ["148 任务的原始报告：", ""]
        for record in reports:
            if record["report"].exists():
                lines.append("- " + link(record["report"], output,
                                         record["run_role"] + "/" + record["variant"] + " .ncu-rep")
                             + "；" + link(record["csv"], output, "原始 CSV"))
        lines.append("")
    lines += ["## 完整数据与复现来源", "",
              link(output / "combined_results.csv", output, "全部已验证测量 CSV") + " · "
              + link(output / "final_comparison.svg", output, "可编辑 SVG 图") + "。", "",
              "以下列出已保存的原始文件及明确标记的控制台恢复文件；"
              "恢复文件不替代遗失的逐次原始数据。本报告生成器只读取文件，不触发 GPU、编译或任何重测。", ""]
    provenance = []
    for run in (initial, refinement):
        for relative in ("request.json", "artifacts/results.json", "artifacts/console_results.json", "console.log",
                         "source_manifest.json", "build.json", "gpu.json", "ncu_metrics.json", "artifacts/ncu/ncu_summary.json"):
            path = run["path"] / relative
            if path.exists():
                provenance.append([run["role"], link(path, output, relative), "`" + sha256(path) + "`"])
        sources = run["path"] / "sources"
        if sources.exists():
            lines += [f"{run['role']} 冻结源代码：" + link(sources, output, str(sources.relative_to(run["path"]))) + "。", ""]
    lines += [table(["会话", "原始文件", "SHA-256"], provenance), "",
              "生成命令：", "", "```bash",
              f"python combine_report.py --initial {os.path.relpath(initial['path'], output)} "
              f"--refinement {os.path.relpath(refinement['path'], output)}", "```", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--refinement", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    # Validate both complete runs before producing or replacing any artifacts.
    initial = load_run(args.initial, "initial")
    refinement = load_run(args.refinement, "refinement")
    add_historical_ratios(initial, refinement)
    ncu_records, ncu_status = [], {}
    for run in (initial, refinement):
        records, status = read_ncu(run)
        ncu_records.extend(records)
        ncu_status[run["role"]] = status
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plot(initial, refinement, output)
    rows = initial["rows"] + refinement["rows"]
    with (output / "combined_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = generate_report(initial, refinement, output, ncu_records, ncu_status)
    (output / "FINAL_REPORT.md").write_text(report)
    print("WROTE", output / "FINAL_REPORT.md", flush=True)
    print("VERIFIED_ROWS", len(rows), "NCU_PROFILES", len(ncu_records), flush=True)


if __name__ == "__main__":
    main()
