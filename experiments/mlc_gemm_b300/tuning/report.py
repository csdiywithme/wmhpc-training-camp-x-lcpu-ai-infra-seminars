"""Generate a source-grounded report from saved B300 tuning measurements.

Usage: python report.py runs/<timestamp> [--no-plot]
Only reads existing measurements; never builds or invokes a GPU workload.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics


PEAK_TFLOPS = 2250.0
PIPELINE_CANDIDATES = ("wide", "full_late", "full_early")
VARIANTS = ("original", "fenced", *PIPELINE_CANDIDATES, "cublas")
COLORS = {
    "original": "#697586", "fenced": "#8b64b5", "wide": "#dfa33a",
    "full_late": "#279589", "full_early": "#2466a8", "cublas": "#c45652",
}


def number(value, digits=2):
    return "—" if value is None else f"{value:,.{digits}f}"


def ratio(numerator, denominator):
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def table(headers, rows):
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows),
    ])


def load_measurements(run):
    result = json.loads((run / "artifacts/results.json").read_text())
    request = result.get("request") or json.loads((run / "request.json").read_text())
    seeds = request.get("seeds", [0, 1, 2])
    workers = int(request.get("sm_count", 148)) // 2
    if workers < 1:
        raise ValueError("The two-CTA scheduler requires at least one worker")
    rows = []
    for requested_case in request["cases"]:
        name = requested_case["name"]
        case = result.get("cases", {}).get(name, {})
        m, n, k = case.get("shape", requested_case["shape"])
        tasks = (m // 512) * (n // 256)
        waves = math.ceil(tasks / workers)
        measurements = case.get("measurements", {})
        for variant in (*requested_case["variants"], "cublas"):
            item = measurements.get(variant, {})
            means = item.get("round_means_ms", [])
            elapsed = item.get("median_ms")
            if elapsed is None and means:
                elapsed = statistics.median(means)
            if elapsed is not None and (not math.isfinite(elapsed) or elapsed <= 0):
                elapsed = None
            validation = case.get("validation", {}).get(variant, [])
            passed_seeds = {v.get("seed") for v in validation if v.get("passed") is True}
            valid_seeds = all(seed in passed_seeds for seed in seeds)
            post = case.get("post_timing_validation", {}).get(variant, {})
            verified = valid_seeds and post.get("passed") is True
            complete = (case.get("status") == "complete" and
                        len(means) == request.get("rounds", 5) and verified)
            cv = (100 * statistics.stdev(means) / statistics.mean(means)
                  if len(means) > 1 and statistics.mean(means) > 0 else None)
            tflops = 2 * m * n * k / elapsed / 1e9 if elapsed else None
            rows.append({
                "case": name, "M": m, "N": n, "K": k,
                "variant": variant, "case_status": case.get("status", "missing"),
                "cluster_tasks": tasks, "cluster_workers": workers, "model_waves": waves,
                "model_task_utilization_percent": 100 * tasks / (workers * waves),
                "gflop": 2 * m * n * k / 1e9,
                "median_ms": elapsed, "tflops": tflops,
                "nominal_peak_percent": tflops / PEAK_TFLOPS * 100 if tflops else None,
                "rounds": len(means), "calls_per_round": item.get("calls_per_round"),
                "min_round_ms": min(means) if means else None,
                "max_round_ms": max(means) if means else None,
                "round_cv_percent": cv,
                "seed_checks_passed": sum(v.get("passed") is True for v in validation),
                "seed_checks_recorded": len(validation), "seed_checks_expected": len(seeds),
                "post_timing_passed": post.get("passed"), "fully_verified": verified,
                "complete_measurement": complete,
                "relative_original": None, "relative_fenced": None,
                "cublas_throughput_percent": None,
            })
    index = {(row["case"], row["variant"]): row for row in rows}
    for row in rows:
        for variant in ("original", "fenced", "cublas"):
            baseline = index.get((row["case"], variant), {})
            value = ratio(baseline.get("median_ms"), row["median_ms"])
            if variant == "cublas":
                row["cublas_throughput_percent"] = value * 100 if value is not None else None
            else:
                row["relative_" + variant] = value
    return result, request, rows, index


def best_for_case(index, name):
    candidates = [index.get((name, variant)) for variant in PIPELINE_CANDIDATES]
    eligible = [row for row in candidates if row and row["complete_measurement"]
                and row["median_ms"] is not None]
    return min(eligible, key=lambda row: row["median_ms"]) if eligible else None


def plot_comparison(run, request, rows, index):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise SystemExit("Plotting requires matplotlib and numpy; install them or use --no-plot") from exc

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"})
    names = [case["name"] for case in request["cases"]]
    positions = np.arange(len(names))
    fig, axes = plt.subplots(3, 1, figsize=(14, 14),
                             gridspec_kw={"height_ratios": [1.4, 1, 1]}, constrained_layout=True)
    ax = axes[0]
    width = 0.125
    for vi, variant in enumerate(VARIANTS):
        values, lower, upper = [], [], []
        for name in names:
            row = index.get((name, variant), {})
            throughput = row.get("tflops") if row.get("complete_measurement") else None
            values.append(throughput if throughput is not None else float("nan"))
            if throughput is not None:
                slow = 2 * row["M"] * row["N"] * row["K"] / row["max_round_ms"] / 1e9
                fast = 2 * row["M"] * row["N"] * row["K"] / row["min_round_ms"] / 1e9
                lower.append(max(0, throughput - slow))
                upper.append(max(0, fast - throughput))
            else:
                lower.append(0)
                upper.append(0)
        ax.bar(positions + (vi - 2.5) * width, values, width, label=variant,
               color=COLORS[variant], yerr=[lower, upper], capsize=1.5,
               error_kw={"elinewidth": 0.7})
    ax.axhline(PEAK_TFLOPS, linestyle="--", linewidth=1, color="#333333",
               label="Nominal dense FP16 peak (2250)")
    ax.set_ylabel("Throughput (TFLOPS)")
    ax.set_title("B300: shape and epilogue experiment", loc="left", fontsize=16, weight="bold")
    ax.set_xticks(positions, names, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(ncol=4, fontsize=9, loc="upper left")

    ax = axes[1]
    for variant in PIPELINE_CANDIDATES:
        values = [index.get((name, variant), {}).get("relative_fenced")
                  if index.get((name, variant), {}).get("complete_measurement") else None
                  for name in names]
        ax.plot(positions, [v if v is not None else float("nan") for v in values],
                marker="o", linewidth=1.6, color=COLORS[variant], label=variant)
    ax.axhline(1, color="#777777", linestyle="--", linewidth=1)
    ax.set_ylabel("Same-shape speedup vs fenced")
    ax.set_xticks(positions, names, rotation=25, ha="right")
    ax.set_title("Pipeline candidates relative to the synchronization control", loc="left")
    ax.legend(ncol=3)
    ax.grid(axis="y", alpha=0.2)

    ax = axes[2]
    sweep = ["rect128", "rect144", "aligned148", "tail152"]
    sweep = [name for name in sweep if (name, "original") in index]
    for variant in ("original", "fenced", "full_early", "cublas"):
        points = [index.get((name, variant)) for name in sweep]
        points = [row for row in points if row and row["complete_measurement"]]
        ax.plot([row["cluster_tasks"] for row in points], [row["median_ms"] * 1000 for row in points],
                marker="o", label=variant, color=COLORS[variant])
    ax.set_xlabel("Logical cluster tasks (M=2048, K=4096)")
    ax.set_ylabel("Latency (microseconds)")
    ax.set_title("Wave boundary: 148 tasks fill two waves; 152 require a third", loc="left")
    ax.set_xticks([index[(name, "original")]["cluster_tasks"] for name in sweep])
    ax.grid(alpha=0.2)
    ax.legend(ncol=4)
    fig.suptitle("Median of round means; bars show range across round means, not confidence intervals",
                 fontsize=10, color="#555555")
    fig.savefig(run / "comparison.png", dpi=170, facecolor="white")
    fig.savefig(run / "comparison.svg", facecolor="white")
    plt.close(fig)


def make_report(run, result, request, rows, index, plotted):
    cases = request["cases"]
    best = [best_for_case(index, case["name"]) for case in cases]
    best = [row for row in best if row]
    winner = max(best, key=lambda row: row["tflops"]) if best else None
    seed_passed = sum(row["seed_checks_passed"] for row in rows)
    seed_recorded = sum(row["seed_checks_recorded"] for row in rows)
    seed_expected = sum(row["seed_checks_expected"] for row in rows)
    post_passed = sum(row["post_timing_passed"] is True for row in rows)
    post_recorded = sum(row["post_timing_passed"] is not None for row in rows)
    env = result.get("environment", {})
    lines = ["# B300：矩阵形状与 v9 写回流水线对照实验", "",
             f"运行目录：`{run.name}`；测量状态：`{result.get('status', 'unknown')}`。",
             "所有数字均读取本目录的 [原始结果](artifacts/results.json)，未复用此前实验的时延。", ""]
    if winner:
        lines += [
            f"完整通过校验的流水线候选中，最高吞吐为 **{winner['variant']} / {winner['case']}**："
            f"{number(winner['median_ms'], 5)} ms、**{number(winner['tflops'])} TFLOPS**，"
            f"为 2250 TFLOPS 标称峰值的 **{number(winner['nominal_peak_percent'])}%**。"
            f"在相同矩阵形状下，它相对 original 为 {number(winner['relative_original'], 3)}×、"
            f"相对 fenced 为 {number(winner['relative_fenced'], 3)}×，"
            f"达到本次 cuBLAS 吞吐的 {number(winner['cublas_throughput_percent'])}%。", ""]
        control = index.get((winner["case"], "fenced"), {})
        gain = ((winner["relative_fenced"] - 1) * 100
                if winner["relative_fenced"] is not None else None)
        fluctuation = max(winner.get("round_cv_percent") or 0,
                          control.get("round_cv_percent") or 0)
        if gain is not None and 0 < gain < fluctuation:
            lines += [
                f"该候选相对 fenced 的吞吐优势仅 **{number(gain)}%**；"
                f"两者 round CV 分别为 {number(winner.get('round_cv_percent'))}% 和 "
                f"{number(control.get('round_cv_percent'))}%。"
                "这点优势处于轮间波动尺度内，**本次未证明确切的流水线收益**。"
                "CV 比较用于说明噪声尺度，并非统计显著性检验。", ""]
    else:
        lines += ["尚无同时完成全部测量与校验的流水线候选，不能给出最佳配置结论。", ""]
    lines += [
        "`fenced` 是补充 TMEM 跨线程同步 fence 的控制组，不计入流水线优化候选。"
        "最佳候选仅在 `wide`、`full_late`、`full_early` 中按本次时延挑选；这是有限候选的事后比较，"
        "不是独立复测确认的最优解。不同矩阵的 TFLOPS 比值表示不同工作负载的吞吐变化，不能当成完成同一任务的加速比。", "",
        "## 测量口径与校验", "",
        f"设备：`{env.get('gpu', request.get('gpu', 'unknown'))}`；"
        f"实际 SM 数：{env.get('sm_count', '未记录')}；编译 SM 数：{request.get('sm_count', 148)}；"
        f"架构：`{env.get('arch', request.get('arch', 'unknown'))}`。",
        "计算 `D[M,N] = A[M,K] @ B[N,K].T`，FP16 输入/输出、FP32 累加。"
        "cuBLAS 通过 `torch.mm(..., out=...)` 调用，关闭 TF32 与 FP16 reduced-precision reduction。", "",
        f"每形状/版本使用 seeds={request.get('seeds', [0, 1, 2])} 完整矩阵验证；"
        f"累计种子验证 **{seed_passed}/{seed_expected} 通过**（已记录 {seed_recorded} 项），"
        f"重复计时后的结果验证 **{post_passed}/{len(rows)} 通过**（已记录 {post_recorded} 项）。"
        "参考为 FP32 GEMM 再舍入 FP16，`rtol=0.02, atol=0.01`；单次校验前将输出填 NaN 检查漏写。", "",
        f"CUDA events 测量完整 GEMM，每版本 {request.get('rounds', 5)} 轮，每轮调用上限 "
        f"{request.get('repeat', '未知')} 次，报告轮均值的中位数。"
        "每次计时前写入 256 MiB 缓冲区，清理在计时之外；版本顺序逐轮打乱。"
        "保持默认动态时钟和功率，不使用 CUDA Graph；事件区间可能包含 host 提交间隙。"
        "round CV 是轮均值的样本标准差除以均值，不能据此直接认定微小差距显著。", "",
        "标称峰值统一采用此前讨论的单卡稠密 FP16 **2250 TFLOPS** 作分母；它不是本卡在测量时钟下的实测峰值。"
        "规格口径参考 [NVIDIA HGX](https://www.nvidia.com/en-us/data-center/hgx/)。", "",
        "## 工作负载与调度模型", "",
        "五种自定义 kernel 都固定启动 **148 个 CTA**，每两个 CTA 组成一个 cluster，即 **74 个 cluster worker**。"
        "一个逻辑任务计算 `512×256` 输出 tile；128、148、152 指逻辑任务总数，grid CTA 数始终不变。"
        "假设所有 worker 同时驻留且每个任务等时，模型利用率为 "
        "`tasks / (74 × ceil(tasks / 74))`。这是解释尾轮的简化模型，**不是 profiler 测得的 SM/Tensor Core 利用率**。", "",
    ]
    shape_rows = []
    for case in cases:
        row = index[(case["name"], "original")]
        shape_rows.append([case["name"], f"{row['M']}×{row['N']}×{row['K']}",
                           row["cluster_tasks"], row["model_waves"],
                           number(row["model_task_utilization_percent"]) + "%", number(row["gflop"])])
    lines += [table(["case", "M×N×K", "逻辑任务", "模型轮数", "模型利用率", "GFLOP"], shape_rows), "",
        "128 个任务为 74+54，148 个任务为 74+74，152 个任务为 74+74+4。"
        "`rect128` 与 `square128` 具有相同 FLOP 和任务数，用于观察长宽比/缓存效应；"
        "`rect144`、`aligned148`、`tail152` 固定 M、K，只跨越 N 方向的两轮边界。"
        "矩阵形状也会改变 A/B 的缓存复用，因此不能将所有吞吐变化归因于尾轮。", "",
        "## 实现与控制变量", "",
        table(["版本", "输入 stage 数", "每 consumer 写回", "累计器复用边界", "每 CTA SMEM 模型"], [
            ["original", 4, "4×64 列", "四次 TMA 读取源 SMEM 完成之后", "225 KiB"],
            ["fenced", 4, "4×64 列", "同 original；增加 TMEM before/after fence", "225 KiB"],
            ["wide", 3, "2×128 列", "两次 TMA 读取源 SMEM 完成之后", "209 KiB"],
            ["full_late", 2, "1×256 列", "单次 TMA 读取源 SMEM 完成之后", "225 KiB"],
            ["full_early", 2, "1×256 列", "TMEM 已全部读出并存入 SMEM 后，TMA store 前", "225 KiB"],
        ]), "",
        "五种自定义 kernel 保留 `512×256×64` 逻辑分块、两个 MMA consumer、384 threads/CTA、"
        "512 列 TMEM 分配和 `l2_group_size=8`。SMEM 模型包含 1 KiB 控制区域："
        "输入每 stage 为 `2×128×64×2 + 128×64×2 = 48 KiB`，"
        "输出为 `2×128×写回列数×2 bytes`；实际编译资源以保存的 NVCC/PTXAS 记录为准。", "",
        "`original→fenced` 隔离同步修补的成本；`fenced→wide/full_late` 同时改变写回粒度和输入流水线深度，"
        "不能当成只减少 store 次数的单因素实验。**`full_late→full_early` 保持其它结构一致，"
        "用于单独测量提前释放累计器、允许下一 tile MMA 与当前 TMA store 重叠的效果。**"
        "这不等于双缓冲输出：两者都等待 TMA 读取源共享内存完成后才复用当前 Dsmem。", "",
        "源码中的 `T.ptx.cp_async.bulk.wait_group(0)` 实际被 TVM lower 为 "
        "`cp.async.bulk.wait_group.read 0`。它等待已提交 TMA store 对源共享内存的读取完成，"
        "让线程能够安全复用该共享内存，**不表示等待全局内存写入全部完成**。"
        "因此不能把 original 的四个 chunk 描述成四次等待显存写入彻底结束。"
        "[PTX read-wait 语义](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-wait-group)", "",
        "## 各形状结果与最佳流水线候选", "",
    ]
    summary = []
    for case in cases:
        name = case["name"]
        chosen = best_for_case(index, name)
        original, fenced, cublas = (index.get((name, v), {}) for v in ("original", "fenced", "cublas"))
        summary.append([name, number(original.get("tflops")), number(fenced.get("tflops")),
                        chosen["variant"] if chosen else "无完整结果",
                        number(chosen["tflops"] if chosen else None), number(cublas.get("tflops")),
                        number(chosen["relative_original"] if chosen else None, 3),
                        number(chosen["relative_fenced"] if chosen else None, 3),
                        number(chosen["nominal_peak_percent"] if chosen else None)])
    lines += [table(["case", "original TF", "fenced TF", "最佳候选", "候选 TF", "cuBLAS TF",
                     "同形状/original ×", "同形状/fenced ×", "候选/标称峰值 %"], summary), "",
              "加速比 >1 表示候选更快，<1 表示更慢。表中的 TF 均指 TFLOPS。", "",
              "## 提前释放累计器的配对消融", ""]
    ablations = []
    for case in cases:
        name = case["name"]
        late, early = (index.get((name, v), {}) for v in ("full_late", "full_early"))
        speed = ratio(late.get("median_ms"), early.get("median_ms"))
        decrease = (1 - early["median_ms"] / late["median_ms"]) * 100 if speed else None
        ablations.append([name, number(late.get("median_ms"), 5), number(early.get("median_ms"), 5),
                          number(speed, 3), number(decrease),
                          number(late.get("round_cv_percent")), number(early.get("round_cv_percent"))])
    lines += [table(["case", "full_late ms", "full_early ms", "late/early ×", "时延降低 %",
                     "late round CV %", "early round CV %"], ablations), "",
              "## 148 个任务附近的形状对照", ""]
    if all((name, "original") in index for name in ("rect144", "aligned148", "tail152")):
        shape_comparison = []
        for variant in VARIANTS:
            a, b, c = (index.get((name, variant), {}) for name in ("rect144", "aligned148", "tail152"))
            shape_comparison.append([variant, number(a.get("median_ms"), 5),
                                     number(b.get("median_ms"), 5), number(c.get("median_ms"), 5),
                                     number(ratio(b.get("tflops"), a.get("tflops")), 3),
                                     number(ratio(c.get("tflops"), b.get("tflops")), 3)])
        lines += [table(["版本", "144 tasks ms", "148 tasks ms", "152 tasks ms",
                         "148/144 吞吐比", "152/148 吞吐比"], shape_comparison), "",
                  "以上三种矩阵的 FLOP 分别随任务数增加；吞吐比不表示相同任务加速。"
                  "152 比148多约2.70% FLOP，但模型多出第三轮；是否出现明显时延跃升，以此表实测为准。", ""]
    else:
        lines += ["本次请求没有完整的144/148/152任务对照组。", ""]
    lines += ["## 固定148任务时改变 K", ""]
    k_names = [case["name"] for case in cases if case["shape"][:2] == [2048, 9472]]
    k_names.sort(key=lambda name: index[(name, "original")]["K"])
    k_rows = []
    for name in k_names:
        for variant in VARIANTS:
            row = index.get((name, variant), {})
            k_rows.append([row.get("K", "—"), variant, number(row.get("median_ms"), 5),
                           number(row.get("tflops")), number(row.get("nominal_peak_percent")),
                           number(row.get("round_cv_percent"))])
    lines += [table(["K", "版本", "ms", "TFLOPS", "标称峰值 %", "round CV %"], k_rows), "",
              "增加 K 不改变输出 tile 数，但增加 K 循环及 FLOP，可能摊薄初始化和写回开销；"
              "也会扩大输入工作集、改变缓存行为。这里比较吞吐，不把更大 K 描述成更快完成同一个 GEMM。", ""]
    if plotted:
        lines += ["## 性能图", "", "![B300 comparison](comparison.png)", "",
                  "图中误差线是各轮均值对应吞吐的最小—最大范围，不是置信区间；未完成全部校验/测量的条目不绘制。"
                  "[SVG 矢量图](comparison.svg)", ""]
    lines += ["## 全部版本数据", ""]
    detailed = []
    for row in rows:
        detailed.append([row["case"], row["variant"], number(row["median_ms"], 5),
                         number(row["tflops"]), number(row["relative_original"], 3),
                         number(row["relative_fenced"], 3), number(row["nominal_peak_percent"]),
                         number(row["cublas_throughput_percent"]), number(row["round_cv_percent"]),
                         f"{row['seed_checks_passed']}/{row['seed_checks_expected']}",
                         "通过" if row["post_timing_passed"] is True else
                         "失败" if row["post_timing_passed"] is False else "缺失",
                         "完整" if row["complete_measurement"] else "不完整"])
    lines += [table(["case", "版本", "ms", "TFLOPS", "/original ×", "/fenced ×", "峰值 %",
                     "cuBLAS %", "round CV %", "种子校验", "计时后校验", "数据状态"], detailed), "",
              "[完整 CSV](measurements.csv) 保存所有版本、模型值、轮数和验证状态；"
              "[原始 JSON](artifacts/results.json) 保存逐次事件时间、每轮均值、运行顺序和遥测。", "",
              "## 来源与运行记录", "",
              "- [请求参数](request.json)、[源码 SHA256](source_manifest.json)",
              "- [原始 v9 源码快照](sources/kernels_advanced.py)、[候选源码快照](sources/kernels_tuned.py)",
              "- [计时代码快照](sources/benchmark.py)、[编译记录](build.json)、[GPU 日志](gpu.json)",
              "- [MLC advanced GEMM](https://mlc.ai/modern-gpu-programming-for-mlsys/zh/chapter_gemm_advanced/index.html)",
              "- [MLC async GEMM](https://mlc.ai/modern-gpu-programming-for-mlsys/zh/chapter_gemm_async/index.html)",
              "- [NVIDIA wave quantization 说明](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#wave-quantization)", ""]
    recovered_path = run / "ncu_metrics.json"
    analysis_path = run / "NCU_ANALYSIS.md"
    if recovered_path.exists() and analysis_path.exists():
        recovered = json.loads(recovered_path.read_text())
        if recovered.get("status") == "recovered_success":
            successful = sum(record.get("profile_returncode") == 0 and
                             record.get("export_returncode") == 0
                             for record in recovered.get("records", []))
            lines += [
                f"**Nsight Compute 的 {successful} 组硬件 profile 与 CSV 导出实际全部成功。**"
                "原始 `ncu_summary.json` 中的 `failed` 是 CSV 解析器误报："
                "解析器期待 long CSV，却收到 `--page raw` 的 wide CSV。"
                "原始 summary 和采集文件保持不变；已从原始 CSV 独立恢复指标，"
                "见 [NCU 分析](NCU_ANALYSIS.md) 与 [恢复后的指标和来源校验值](ncu_metrics.json)。"
                "Profiler 回放计时与正式 benchmark 分开，不能替代重复计时排名。", ""]
    profile_files = [path for path in sorted((run / "artifacts").rglob("*")) if path.is_file()
                     and any(token in path.name.lower() for token in ("ncu", "profile", "summary"))
                     and path.suffix.lower() in (".json", ".csv", ".txt", ".md", ".log", ".ncu-rep")]
    if profile_files:
        lines += ["Profiler 原始附件如下；逐项指标解释参见上面的独立分析（如存在）。"
                  "调度模型利用率不替代 profiler 指标。", ""]
        lines += [f"- [{path.relative_to(run)}]({path.relative_to(run).as_posix()})" for path in profile_files]
        lines += [""]
    else:
        lines += ["此运行目录未发现 profiler 摘要附件；仅凭总时延不能精确拆分搬运、同步与计算开销。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Run directory containing artifacts/results.json")
    parser.add_argument("--no-plot", action="store_true", help="Write Markdown and CSV without matplotlib")
    args = parser.parse_args()
    run = args.run.resolve()
    result, request, rows, index = load_measurements(run)
    with (run / "measurements.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    if not args.no_plot:
        plot_comparison(run, request, rows, index)
    (run / "REPORT.md").write_text(make_report(
        run, result, request, rows, index,
        plotted=(run / "comparison.png").exists() and (run / "comparison.svg").exists()),
        encoding="utf-8")
    print(f"Wrote {run / 'REPORT.md'}")
    print(f"Wrote {run / 'measurements.csv'}")
    if not args.no_plot:
        print(f"Wrote {run / 'comparison.png'} and comparison.svg")


if __name__ == "__main__":
    main()
