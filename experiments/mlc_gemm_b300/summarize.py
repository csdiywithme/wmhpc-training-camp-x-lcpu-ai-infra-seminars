"""Create a Chinese Markdown report and CSV from a saved local B300 run.

Usage: python summarize.py runs/20260918T000000Z [--output report_directory]
Only the Python standard library is required; this never launches GPU work.
Incomplete runs remain explicitly incomplete, with unmeasured cells left blank.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import statistics


VERSIONS = tuple(range(1, 10)) + (0,)
LABELS = {
    1: "同步加载 + MMA（全矩阵适配）",
    2: "K 循环在 TMEM 累加（全矩阵适配）",
    3: "空间分块、多 CTA",
    4: "TMA 异步加载",
    5: "软件流水线",
    6: "持久化 kernel",
    7: "Warp specialization",
    8: "双 CTA cluster",
    9: "多消费者",
    0: "cuBLAS / torch.mm",
}
FIELDS = (
    "version", "technique", "median_of_round_means_ms", "tflops",
    "speedup_v1", "percent_cublas_throughput", "min_round_mean_ms",
    "max_round_mean_ms", "round_cv_percent", "round_count",
    "calls_per_round", "sample_count", "validation_passed",
    "validated_seeds", "max_abs_error", "max_rms_error",
    "max_relative_l2_error", "mismatch_count",
)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def fmt(value, digits=4):
    return f"{value:.{digits}f}" if finite(value) else "—"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def link(path, output, label):
    return f"[{label}](<{os.path.relpath(path, output)}>)"


def rows_for(result):
    measurements = result.get("measurements", {})
    validation = result.get("validation", {})
    expected_seeds = result.get("protocol", {}).get("seeds", [0, 1, 2])
    shape = result.get("protocol", {}).get("shape", [result.get("request", {}).get("size")] * 3)
    operations = 2 * math.prod(shape) if all(finite(x) and x > 0 for x in shape) else None
    rows = []
    for version in VERSIONS:
        item = measurements.get(str(version), {})
        means = item.get("round_means_ms", [])
        means = [x for x in means if finite(x) and x > 0]
        latency = statistics.median(means) if means else None
        checks = validation.get(str(version), [])
        seeds = sorted({x["seed"] for x in checks if "seed" in x})
        passed = bool(checks) and all(x.get("passed") is True for x in checks)
        passed = passed and set(seeds) == set(expected_seeds)

        def maximum(key):
            values = [x[key] for x in checks if finite(x.get(key))]
            return max(values) if values else None

        rows.append({
            "version": f"v{version}" if version else "cuBLAS",
            "technique": LABELS[version],
            "median_of_round_means_ms": latency,
            "tflops": operations / latency / 1e9 if latency and operations else None,
            "speedup_v1": None,
            "percent_cublas_throughput": None,
            "min_round_mean_ms": min(means) if means else None,
            "max_round_mean_ms": max(means) if means else None,
            "round_cv_percent": (100 * statistics.stdev(means) / statistics.mean(means)
                                 if len(means) > 1 else None),
            "round_count": len(means),
            "calls_per_round": item.get("calls_per_round"),
            "sample_count": sum(len(x) for x in item.get("samples_ms", [])),
            "validation_passed": passed,
            "validated_seeds": ",".join(str(x) for x in seeds),
            "max_abs_error": maximum("max_abs_error"),
            "max_rms_error": maximum("rms_error"),
            "max_relative_l2_error": maximum("relative_l2_error"),
            "mismatch_count": sum(x.get("mismatch_count", 0) for x in checks) if checks else None,
        })
    baseline = rows[0]["median_of_round_means_ms"]
    cublas = rows[-1]["median_of_round_means_ms"]
    for row in rows:
        latency = row["median_of_round_means_ms"]
        if latency:
            row["speedup_v1"] = baseline / latency if baseline else None
            row["percent_cublas_throughput"] = 100 * cublas / latency if cublas else None
    return rows


def environment_lines(result):
    env = result.get("environment", {})
    req = result.get("request", {})
    cap = ".".join(str(x) for x in env.get("compute_capability", [])) or "未记录"
    memory = env.get("memory_bytes")
    memory_text = f"{memory / 2**30:.2f} GiB ({memory} bytes)" if finite(memory) else "未记录"
    lines = [
        f"设备：{env.get('gpu', '未记录')}；SM 数：{env.get('sm_count', '未记录')}；"
        f"compute capability：{cap}；显存：{memory_text}。",
        f"运行平台：Modal；请求 GPU：{req.get('gpu', '未记录')}；"
        f"编译目标：`{req.get('arch', '未记录')}`；CUDA stream：`{env.get('cuda_stream', '未记录')}`。",
        f"PyTorch CUDA：`{env.get('torch_cuda', '未记录')}`；时钟策略：{env.get('clock_policy', '未记录')}。",
    ]
    packages = env.get("packages", {})
    if packages:
        lines.append("软件版本：" + "；".join(f"`{key}={value if value is not None else '未查得'}`"
                                           for key, value in packages.items()) + "。")
    cublas = env.get("cublas", {})
    if cublas:
        lines.append("cuBLAS 配置：`" + json.dumps(cublas, ensure_ascii=False, sort_keys=True) + "`。")
    telemetry = result.get("telemetry", [])
    if telemetry:
        lines.extend(["", "遥测原始字段依次为 GPU 名称、UUID、驱动版本、P-state、温度、功耗、"
                      "功率上限、SM 时钟、显存时钟、GPU 利用率、已用显存。它们是轮次间快照，"
                      "不能代表每次 kernel 执行时的时钟。", "", "```text"])
        for item in telemetry:
            lines.append(f"{item.get('time_utc', '未记录')} | rc={item.get('returncode', '未记录')} | "
                         f"{item.get('stdout', '').strip()}")
            if item.get("stderr"):
                lines.append("stderr: " + item["stderr"].strip())
        lines.append("```")
    return lines


def build_details(run, output):
    """Read saved compiler diagnostics; never conflate preflight with profiling."""
    build_path = run / "build.json"
    if not build_path.exists() and (run / "reused_build.txt").exists():
        build_path = Path((run / "reused_build.txt").read_text().strip()) / "build.json"
    if not build_path.exists():
        return [], False
    built = read_json(build_path)
    metadata = []
    compiler = None
    for record in built.get("records", []):
        command = record.get("command", [])
        if command[:2] == ["nvcc", "--version"]:
            compiler = record.get("stdout", "").strip()
        if "compile_one.py" not in command:
            continue
        try:
            metadata.append(json.loads(record.get("stdout", "")))
        except json.JSONDecodeError:
            continue
    lines = ["", "## 编译记录", "", "来源：" + link(build_path, output, "CPU 编译日志") + "。"]
    if compiler:
        lines.extend(["", "```text", compiler, "```"])
    revisions = sorted({item["upstream_revision"] for item in metadata if item.get("upstream_revision")})
    if revisions:
        lines.extend(["", "教程源码 revision：" + "、".join(f"`{rev}`" for rev in revisions) + "。"])
    preflights = []
    for item in metadata:
        preflight = item.get("nvcc_preflight", {})
        diagnostic = preflight.get("stderr", "")
        regs = re.search(r"Used (\d+) registers", diagnostic)
        spill = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", diagnostic)
        if regs and spill:
            preflights.append((item["version"], int(regs.group(1)), *(int(x) for x in spill.groups())))
    if preflights:
        lines.extend(["", "以下是 NVCC 对生成 CUDA 源码做的 CPU 端 `sm_103a` cubin 预编译资源报告。"
                      "实际共享库的 CUDA 编译在 GPU 端加载模块时完成、位于计时之外；这些数值并非对运行中"
                      "已加载机器码的 profiler 采样，也不是实际访存流量。", "",
                      "| 版本 | 每线程寄存器数 | stack frame B | spill stores B | spill loads B |",
                      "|---|---:|---:|---:|---:|"])
        for version, registers, stack, stores, loads in sorted(preflights):
            lines.append(f"| v{version} | {registers} | {stack} | {stores} | {loads} |")
    v1_spills = any(version == 1 and (stores or loads)
                    for version, registers, stack, stores, loads in preflights)
    return lines, v1_spills


def report(result, rows, run, source, output):
    req = result.get("request", {})
    protocol = result.get("protocol", {})
    requested_rounds = req.get("rounds", 0)
    compiler_lines, v1_spills = build_details(run, output)
    spill_explanation = (
        "CPU 预编译的 ptxas 资源报告已经显示 v1 存在 spill（见上表），与适配层较高的寄存器压力一致。"
        "这些静态数值不能直接当成运行时 spill 流量或性能归因。" if v1_spills else
        "较高寄存器压力可能造成 spill；本报告未取得相关编译证据，不将 spill 写成已证实的结论。")
    complete = (result.get("status") == "complete"
                and all(row["validation_passed"] and row["round_count"] == requested_rounds
                        and row["median_of_round_means_ms"] is not None for row in rows))
    shape = protocol.get("shape", [req.get("size", "?")] * 3)
    lines = ["# Modal B300：教程九个 GEMM 版本实测", "",
             f"运行目录：`{run.name}`。原始状态：`{result.get('status', 'unknown')}`。",
             "九个版本及 cuBLAS 均完成正确性检查和计时。" if complete else
             "**本报告不完整：未取得全部版本的完整正确性与计时结果，缺失项以“—”显示。**",
             "", f"运算为 `{protocol.get('operation', 'D = A @ B.T')}`，"
             f"`M×N×K = {'×'.join(str(x) for x in shape)}`，{protocol.get('dtype', '精度未记录')}。",
             "延迟取各轮平均延迟的中位数；TFLOPS 按 `2MNK / 时间` 计算。cuBLAS 百分比"
             "表示吞吐比（cuBLAS 延迟 / 当前延迟），超过 100% 表示本次测量更快。", "",
             "| 版本 | 优化步骤 | 延迟 ms | TFLOPS | 相对 v1 | cuBLAS 吞吐比 | 轮均值范围 ms | 轮间 CV | 样本数 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        rounds = row["round_count"]
        count = row["sample_count"]
        lines.append(f"| {row['version']} | {row['technique']} | {fmt(row['median_of_round_means_ms'])} | "
                     f"{fmt(row['tflops'], 2)} | {fmt(row['speedup_v1'], 2)}× | "
                     f"{fmt(row['percent_cublas_throughput'], 2)}% | "
                     f"{fmt(row['min_round_mean_ms'])}–{fmt(row['max_round_mean_ms'])} | "
                     f"{fmt(row['round_cv_percent'], 2)}% | {count}（{rounds} 轮） |")
    if complete:
        best = min(rows[:-1], key=lambda row: row["median_of_round_means_ms"])
        lines.extend(["", f"本次九个实现中，{best['version']} 的延迟最低："
                      f"{fmt(best['median_of_round_means_ms'])} ms，"
                      f"{fmt(best['tflops'], 2)} TFLOPS，为 cuBLAS 吞吐的 "
                      f"{fmt(best['percent_cublas_throughput'], 2)}%。接近的结果需结合轮间波动解释。"])
        ms = {row["version"]: row["median_of_round_means_ms"] for row in rows}
        lines.extend(["", "这次运行中的主要变化：", "",
                      f"- v2→v3：多 CTA 空间分块后快 {ms['v2']/ms['v3']:.2f}×。",
                      f"- v3→v4：吞吐仅提升 {(ms['v3']/ms['v4']-1)*100:.2f}%；当前源码没有复现原表这一段的巨大差距。",
                      f"- v4→v5：软件流水线后快 {ms['v4']/ms['v5']:.2f}×。",
                      f"- v6→v7：warp specialization 的延迟反而增加 {(ms['v7']/ms['v6']-1)*100:.2f}%。这是本次参数下的观测，尚未用 profiler 定位原因。",
                      f"- v7→v8：双 CTA cluster 后快 {ms['v7']/ms['v8']:.2f}×；v8→v9 再快 {ms['v8']/ms['v9']:.2f}×。",
                      "", "![性能比较](performance.png)"])
    lines.extend(["", "## 正确性", "",
                  f"参考：{protocol.get('reference', '未记录')}。逐元素容差为 "
                  f"`abs(actual−reference) ≤ {protocol.get('atol', '?')} + "
                  f"{protocol.get('rtol', '?')} × abs(reference)`。"
                  "每次检查先将输出填为 NaN，再检查完整矩阵及有限值。",
                  f"共同随机种子：`{protocol.get('seeds', [])}`。下表误差为各个种子中的最大值，"
                  "失败元素数为所有检查之和。", "",
                  "| 版本 | 种子 | 检查 | 最大绝对误差 | 最大 RMS 误差 | 最大相对 L2 误差 | 失败元素数 |",
                  "|---|---|---|---:|---:|---:|---:|"])
    for row in rows:
        lines.append(f"| {row['version']} | {row['validated_seeds'] or '—'} | "
                     f"{'通过' if row['validation_passed'] else '失败或未完成'} | "
                     f"{fmt(row['max_abs_error'], 6)} | {fmt(row['max_rms_error'], 6)} | "
                     f"{fmt(row['max_relative_l2_error'], 8)} | "
                     f"{row['mismatch_count'] if row['mismatch_count'] is not None else '—'} |")
    lines.extend(["", "## 测量条件", "",
                  f"计时器：{protocol.get('timer', '未记录')}。每个区间覆盖完整 GEMM，"
                  "CUDA event 区间可能包含主机提交空隙，因此不等同于 profiler 的 kernel 指令执行时间。",
                  f"缓存策略：{protocol.get('cache', '未记录')}。该策略用于减少输入热缓存影响，"
                  "未直接验证每条 cache line 均被驱逐。",
                  f"预热：{protocol.get('warmup', '未记录')}。每轮次数：{protocol.get('repeat', '未记录')}。",
                  f"排除项：{', '.join(protocol.get('excludes', [])) or '未记录'}。"
                  f"输入 strides：`{json.dumps(protocol.get('input_strides', {}))}`。",
                  f"各轮运行顺序（0 为 cuBLAS）：`{json.dumps(result.get('round_order', []))}`。",
                  f"调优策略：{protocol.get('tuning', '未记录')}。", "",
                  "## 环境与遥测", "", *environment_lines(result), *compiler_lines, "",
                  "## 解释范围", "",
                  "v1 原例只计算 128×128×64，v2 原例只计算一个输出 tile。这里为取得相同完整"
                  "矩阵工作量，让一个 CTA 串行遍历全部输出 tile；v1 还将独立 K=64 MMA 的 TMEM "
                  "部分结果读回并在 FP32 寄存器求和。它们是明确标注的全矩阵适配，不能当成原样运行教程小例子。",
                  "v1 同时保留 Dreg[128] 与 Dsum[128] 的 FP32 局部数组。" + spill_explanation +
                  "v1→v2 差异还包含适配层的 TMEM 读回、寄存器归约和寄存器压力变化，不能将全部加速归因于 K 循环。",
                  "这组结果只对应本次 B300、给定形状、默认动态时钟和缓存策略。教程的 B200 结果使用不同"
                  "设备与锁频条件，不能直接作相同条件的速度对比；也不据此声称达到 B300 峰值。", "",
                  "## 复现材料", "",
                  "- " + link(source, output, "完整结果 JSON（逐次延迟、轮均值、误差与遥测）"),
                  "- " + link(output / "results.csv", output, "汇总 CSV")])
    lines.append("- 教程来源：[基础 GEMM](https://mlc.ai/modern-gpu-programming-for-mlsys/zh/chapter_gemm_basics/index.html)、"
                 "[异步 GEMM](https://mlc.ai/modern-gpu-programming-for-mlsys/zh/chapter_gemm_async/index.html)、"
                 "[进阶 GEMM](https://mlc.ai/modern-gpu-programming-for-mlsys/zh/chapter_gemm_advanced/index.html)")
    for name, label in (("source_manifest.json", "本次源码 SHA-256"),
                        ("request.json", "运行参数"), ("build.json", "编译日志"),
                        ("gpu.json", "GPU 运行日志"), ("build.tar.gz", "编译产物归档"),
                        ("sources/kernels_basics.py", "v1/v2 适配与局部数组源码"),
                        ("sources/benchmark.py", "测量脚本"),
                        ("reused_build.txt", "复用编译记录")):
        path = run / name
        if path.exists():
            lines.append("- " + link(path, output, label))
    hashes = result.get("build_sha256", {})
    if hashes:
        lines.extend(["", "已加载共享库 SHA-256：", "", "```text"])
        lines.extend(f"{name}  {digest}" for name, digest in sorted(hashes.items()))
        lines.append("```")
    if result.get("error"):
        lines.extend(["", "保存的失败信息：", "", "```text", result["error"].rstrip(), "```"])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Run directory containing artifacts/results.json")
    parser.add_argument("--output", type=Path, help="Report directory (default: run directory)")
    args = parser.parse_args()
    run = args.run.resolve()
    source = run / "artifacts" / "results.json"
    if not source.is_file():
        parser.error(f"Saved result does not exist yet: {source}")
    result = read_json(source)
    output = (args.output or run).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = rows_for(result)
    with (output / "results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (output / "RESULTS.md").write_text(report(result, rows, run, source, output), encoding="utf-8")
    print(output / "RESULTS.md")
    print(output / "results.csv")


if __name__ == "__main__":
    main()
