"""Generate a report from saved measurements; no GPU work or new timing."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    path = args.run.resolve()
    raw = path / "artifacts/results.json"
    result = json.loads(raw.read_text())
    assert result["status"] == "complete" and len(result["cases"]) == 3
    lines = ["# 当前最高记录内核：形状、tile 与任务轮数", "",
             "本次固定使用既有 `hgemm_v9_wide`，对三个邻近形状做新的稳态配对实验。",
             "FP16 输入/输出、FP32 累加，禁止 FP16 中间归约；连续复用缓冲区，不显式清缓存。",
             "两种实现同卡交错运行，每轮各计时 150 次；正式结果为七轮均值的中位数。", ""]
    session = result["sessions"][0]
    lines += [f"设备：{session['gpu']}，{session['sm_count']} SM；"
              f"PyTorch {session['packages']['torch']}，cuBLAS {session['packages']['nvidia-cublas']}。", ""]
    cases = result["cases"]
    best_name = max(cases, key=lambda name: cases[name]["measurements"]["custom_wide"]["tflops"])
    best = cases[best_name]
    item = best["measurements"]["custom_wide"]
    lines += [f"这三个形状中，自定义版本最高记录为 **{item['tflops']:,.2f} TFLOPS**，"
              f"来自 `{'×'.join(map(str,best['spec']['shape']))}`，时延 **{item['median_ms']*1000:.3f} μs**，"
              f"为 2250 TFLOPS 标称峰值的 **{item['peak_percent']:.2f}%**。", "",
              "标称峰值依据 [NVIDIA HGX 官方规格](https://www.nvidia.com/en-us/data-center/hgx/)"
              "（2026-09-20 核对）：HGX B300 为 8 卡，FP16/BF16 Tensor Core 合计 36 PFLOPS；"
              "脚注注明该值为 sparse，dense 为其一半。因此单卡 dense 为"
              " `36×1000÷8÷2 = 2250 TFLOPS`，单卡 sparse 为 4500 TFLOPS。"
              "本实验为 dense GEMM，百分比使用 dense 标称值，未经运行中频率修正，"
              "也不是硬件计数器测得的 Tensor Core 利用率。", "",
              "## 有效 tile 与调度", "",
              "基本分块 `BLK_M=128, BLK_N=128` 不是调度器分配的完整输出块。"
              "两个 CTA、两个 MMA consumer 合起来处理 **512×256** 输出；K tile 为 64。",
              "固定启动 148 个 CTA，每两个组成一个 cluster，共 **74 个 worker**。", "",
              "`任务数 = (M/512) × (N/256)`；`最长任务序列 = ceil(任务数/74)`。",
              "每个 worker 的任务编号从自身 ID 开始，每处理一块加 74。"
              "K=8192 时，每个输出任务执行 128 个 K 分块。", "",
              "| M×N×K | 任务 | 各 worker 工作量 | 逻辑轮数 | 槽位利用率模型 |",
              "| --- | ---: | --- | ---: | ---: |"]
    for name, case in cases.items():
        spec = case["spec"]
        g = spec["geometry"]
        counts = {n: g["tasks_by_worker"].count(n) for n in sorted(set(g["tasks_by_worker"]))}
        allocation = " + ".join(f"{workers} 个做 {tasks} 块" for tasks, workers in counts.items())
        lines.append(f"| {'×'.join(map(str,spec['shape']))} | {g['logical_tasks']} | {allocation} | "
                     f"{g['logical_rounds']} | {g['model_slot_utilization_percent']:.2f}% |")
    lines += ["", "这里的轮数是 persistent kernel 内部逻辑工作量，不是额外 kernel launch，也不要求轮间全局同步。"
              "槽位利用率是等时任务模型，不等于实测 SM 活跃率或 Tensor Core 峰值比例。", "",
              "## 新稳态实测", "",
              "| N（M=2048,K=8192） | wide μs | wide TFLOPS | cuBLASLt μs | cuBLASLt TFLOPS | wide/库吞吐 | wide CV | 库 CV |",
              "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    rows = []
    for name, case in cases.items():
        wide, lib = (case["measurements"][key] for key in ("custom_wide", "lt_tuned"))
        ratio = lib["median_ms"] / wide["median_ms"]
        lines.append(f"| {case['spec']['shape'][1]} | {wide['median_ms']*1000:.3f} | {wide['tflops']:,.2f} | "
                     f"{lib['median_ms']*1000:.3f} | {lib['tflops']:,.2f} | {100*ratio:.2f}% | "
                     f"{wide['round_cv_percent']:.2f}% | {lib['round_cv_percent']:.2f}% |")
        for method, item in case["measurements"].items():
            rows.append({"case": name, "method": method, "M": 2048, "N": case["spec"]["shape"][1], "K": 8192,
                         "tasks": case["spec"]["geometry"]["logical_tasks"],
                         "logical_rounds": case["spec"]["geometry"]["logical_rounds"],
                         "median_ms": item["median_ms"], "tflops": item["tflops"],
                         "peak_percent": item["peak_percent"], "round_cv_percent": item["round_cv_percent"]})
    lines += ["", "中心形状的库实现复用上次已选定的 cuBLASLt 配置，未重新搜索；它不一定是稳态协议下的库全局最优。"
              "两个端点为 K=8192 的新形状，各进行一次候选搜索。所有结果是本次受限候选集合与运行条件下的记录。", "",
              "## 运行顺序敏感性", "",
              "本次观察到较明显的先后顺序相关波动，不能将中心形状的小幅中位数优势表述为稳定超过 cuBLASLt。"
              "以下是对已有七轮样本的事后分层分析，没有新增 GPU 计时。顺序由冻结 benchmark.py 的"
              "固定随机种子 `20260920+round_index` 和方法插入顺序重建。"
              "wide 在第 0/2/3/5 轮先测，共四轮；库在第 1/4/6 轮先测，共三轮。"
              "中心形状七轮均为先测的方法更快。下表同一行的两个中位数来自不同轮次，不能当成逐轮配对结果。", "",
              "| 中心形状中的测量位置 | wide 轮均值中位数 μs | 库轮均值中位数 μs | wide/库吞吐比 |",
              "| --- | ---: | ---: | ---: | ---: |"]
    orders = []
    for round_index in range(result["request"]["rounds"]):
        order = ["custom_wide", "lt_tuned"]
        random.Random(20260920 + round_index).shuffle(order)
        orders.append(order)
    center = cases["aligned148_k8192"]["measurements"]
    for position, label in ((0, "各自在轮次中先测"), (1, "各自在轮次中后测")):
        values = {method: statistics.median([item["round_means_ms"][i]
                   for i, order in enumerate(orders) if order[position] == method])
                  for method, item in center.items()}
        ratio = values["lt_tuned"] / values["custom_wide"]
        lines.append(f"| {label} | {values['custom_wide']*1000:.3f} | "
                     f"{values['lt_tuned']*1000:.3f} | {ratio*100:.2f}% |")
    lines += ["", "该分层比较也不是独立的新对照实验，不能作为修正后的确定加速比。"
              "它表明混合位置得到的几个百分点差距不稳健；频率/温度/功率快照不足以确认顺序效应的单一成因。", "",
              "## 148→152 的变化", "",
              "输出任务与 FLOPs 增加 `152/148−1 = 2.70%`；自定义版本最长工作序列由两块变成三块。"
              "等时任务模型下最长序列增加 50%，实际时延还受启动/排空、缓存与动态时钟影响。", ""]
    tail = cases["tail152_k8192"]["measurements"]
    for method, label in (("custom_wide", "wide"), ("lt_tuned", "cuBLASLt")):
        time_change = 100 * (tail[method]["median_ms"] / center[method]["median_ms"] - 1)
        throughput_change = 100 * (tail[method]["tflops"] / center[method]["tflops"] - 1)
        lines += [f"- {label}：时延变化 **{time_change:+.2f}%**，吞吐变化 **{throughput_change:+.2f}%**。"]
    lines += ["", "cuBLASLt 有自己的 tile/cluster 配置，不能把自定义版本的 74 worker、512×256 tile 模型直接套在库上。"
              "本次未采集 NCU，任务模型来自自定义源代码，不能将推算槽位利用率称为硬件计数器结果。", "",
              "本次 cuBLASLt 中心与152端点选中的 tile ID 分别为 "
              f"{cases['aligned148_k8192']['selection']['config']['tile_id']['value']} 和 "
              f"{cases['tail152_k8192']['selection']['config']['tile_id']['value']}，确实采用了不同 tile 配置；"
              "这些记录只能解释本次库路径，不反推早先默认 torch.mm 的具体 kernel。", "",
              "## 与旧实验的关系", "",
              "同一中心形状旧清缓存记录：wide 为 **1656.41 TFLOPS /191.878 μs**，"
              "其同会话默认库为 1641.83 TFLOPS。后续另一会话的调优 cuBLASLt 清缓存记录为 1748.05 TFLOPS。",
              "这些旧记录全部复用，没有重新运行；不能用新稳态值与旧清缓存值计算代码优化收益。"
              "本次没有改变 wide 的 tile 或流水线，也没有重新测试其他自定义版本。", "",
              "## 校验与审计", ""]
    audit = {"status": "passed", "cases": len(cases), "sessions": len(result["sessions"]),
             "candidate_checks": 0, "seed_checks": 0, "post_checks": 0,
             "formal_samples": 0, "tuning_samples": 0,
             "results_sha256": hashlib.sha256(raw.read_bytes()).hexdigest()}
    for name, case in cases.items():
        assert case["status"] == "complete"
        for candidate in case["candidates"].values():
            assert candidate["validation"]["passed"]
            audit["candidate_checks"] += 1
            audit["tuning_samples"] += sum(len(x) for x in candidate["samples_ms"])
        for method, item in case["measurements"].items():
            records = case["validation"][method]
            assert len(records) == 3 and all(v["passed"] for v in records)
            assert item["post_validation"]["passed"]
            audit["seed_checks"] += len(records)
            audit["post_checks"] += 1
            means = [statistics.mean(x) for x in item["samples_ms"]]
            assert len(means) == 7 and means == item["round_means_ms"]
            assert statistics.median(means) == item["median_ms"]
            assert all(0 < x < float("inf") for samples in item["samples_ms"] for x in samples)
            audit["formal_samples"] += sum(len(x) for x in item["samples_ms"])
    for name, digest in json.loads((path / "source_manifest.json").read_text()).items():
        assert hashlib.sha256((path / "sources" / name).read_bytes()).hexdigest() == digest
    build = json.loads((path / "build/build.json").read_text())
    assert hashlib.sha256((path / "build/libgemmbench.so").read_bytes()).hexdigest() == build["shared_object_sha256"]
    audit["build_artifacts_checked"] = 1
    for metadata in build["custom_builds"].values():
        for name, digest in metadata["sha256"].items():
            assert hashlib.sha256((path / "build" / name).read_bytes()).hexdigest() == digest
            audit["build_artifacts_checked"] += 1
    assert cases["aligned148_k8192"]["reused_lt_selection"]["no_search_timing"]
    assert all(not v["samples_ms"] for v in cases["aligned148_k8192"]["candidates"].values())
    lines += [f"{audit['candidate_checks']} 个 Lt 候选输出校验、{audit['seed_checks']} 次正式方法种子校验、"
              f"{audit['post_checks']} 次计时后校验全部通过。保存 {audit['formal_samples']} 个正式计时样本"
              f"和 {audit['tuning_samples']} 个端点调优样本。原始样本复算轮均值与中位数一致，源码及"
              f" {audit['build_artifacts_checked']} 个编译产物的哈希一致。", "",
              "默认动态频率和功率，不使用 CUDA Graph；频率/功率数据是轮间快照，不能据此确定单次 kernel 的平均频率或限频原因。", "",
              f"[原始结果]({path}/artifacts/results.json) · [汇总 CSV]({path}/summary.csv) · [审计]({path}/audit.json)", ""]
    (path / "RESULTS.md").write_text("\n".join(lines))
    (path / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    with (path / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(path / "RESULTS.md")
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
