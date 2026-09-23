"""Read-only report generator; never launches or repeats a GPU experiment."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    run = args.run.resolve()
    result = json.loads((run / "artifacts/results.json").read_text())
    old_path = root.parent / "tuning/runs/20260918T092500Z/artifacts/results.json"
    old = json.loads(old_path.read_text())
    name_map = {"square4096": "square128", "aligned148": "aligned148",
                "tail152": "tail152", "aligned148_k8192": "aligned148_k8192"}
    labels = {"lt_tuned": "cuBLASLt tuned", "gemmex_autotune": "GemmEx AUTOTUNE",
              "torch_default": "torch.mm default"}
    lines = ["# B300 cuBLAS：T1–T3 实测", "",
             f"运行：`{run.name}`；状态：`{result['status']}`。仅执行 T1、T2、T3。", "",
             "FP16 输入/输出、FP32 累加，禁止 FP16 中间归约。吞吐按 `2MNK / 时间` 计算；",
             "理论峰值分母为 2250 TFLOPS。选算法与正式计时使用独立样本。", ""]
    all_rows = []
    for case in result["request"]["cases"]:
        data = result["cases"].get(case["name"], {})
        for method, item in data.get("measurements", {}).items():
            if "tflops" not in item:
                continue
            row = {"case": case["name"], "M": case["shape"][0], "N": case["shape"][1],
                   "K": case["shape"][2], "method": method, "cache": case["cache"],
                   **{k: item[k] for k in ("median_ms", "tflops", "peak_percent", "round_cv_percent")},
                   "selected_candidate": data["selected_candidate"] if method == "lt_tuned" else "",
                   "sessions": json.dumps(data.get("sessions_used", [])),
                   "all_seeds_passed": all(v["passed"] for v in data["validation"][method]),
                   "post_passed": item["post_validation"]["passed"]}
            all_rows.append(row)
    if all_rows:
        best = max(all_rows, key=lambda row: row["tflops"])
        lines += [f"最高记录：**{best['tflops']:,.2f} TFLOPS（标称峰值 {best['peak_percent']:.2f}%）**，"
                  f"来自 {best['case']} / {labels[best['method']]}，时延 {best['median_ms']:.6f} ms。", ""]
    lines += ["本次有界算法搜索及大方阵扫描**没有达到标称峰值的 90%**。",
              "AUTOTUNE 与手动 Lt 调优在两个矩形上的差距小于 0.5%，与轮间波动同量级。",
              "大方阵未呈现越大吞吐越高；调优所选配置也并非在正式稳态计时中始终优于默认库。", ""]
    lines += ["## T1：cuBLASLt 候选调优", "",
              "32 个请求候选，64 MiB workspace；筛除不符合精度约束或正确性失败的配置。",
              "逐次在计时外清理 256 MiB 缓冲区。历史默认库结果直接复用，未重跑；",
              "**历史比值来自不同 GPU 会话，不用于认定几个百分点的因果收益。**", "",
              "此处统一引用后续形状实验的历史基线。4096³ 在更早的九版本实验中还记录过"
              "1530.23 TFLOPS，本次 1541.05 TFLOPS 相对该记录仅高约 0.7%；"
              "两份旧结果都保留，不把选用某一次历史值当成调优收益的证明。", "",
              "| M×N×K | 调优后 TFLOPS | 延迟 μs | 轮间 CV | 历史默认 TFLOPS | 历史吞吐比 |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in all_rows:
        if row["case"] in name_map and row["method"] == "lt_tuned":
            previous = old["cases"][name_map[row["case"]]]["measurements"]["cublas"]
            lines.append(f"| {row['M']}×{row['N']}×{row['K']} | {row['tflops']:,.2f} | "
                         f"{row['median_ms']*1000:.3f} | {row['round_cv_percent']:.2f}% | "
                         f"{previous['tflops']:,.2f} | {row['tflops']/previous['tflops']:.4f}× |")
    lines += ["", "## T2：内置 AUTOTUNE", "",
              "首次内部搜索单独执行，排除在正式计时之外；复用同一 handle 的缓存。",
              "两种新库路径在同一形状、同一缓存协议下交错计时。", "",
              "| 形状 | 方法 | TFLOPS | 延迟 μs | 轮间 CV |", "| --- | --- | ---: | ---: | ---: |"]
    for row in all_rows:
        if row["case"] in ("aligned148", "tail152"):
            lines.append(f"| {row['case']} | {labels[row['method']]} | {row['tflops']:,.2f} | "
                         f"{row['median_ms']*1000:.3f} | {row['round_cv_percent']:.2f}% |")
    lines += ["", "## T3：新大矩阵稳态性能", "",
              "连续使用相同缓冲区，无显式缓存清理；不代表整个工作集驻留 L2。",
              "默认 torch.mm 与调优后 cuBLASLt 在同一会话交错计时。",
              "此表不能与 T1/T2 的清缓存结果直接作受控加速比较。", "",
              "| M=N=K | 方法 | TFLOPS | 峰值比例 | 延迟 ms | 轮间 CV |", "| ---: | --- | ---: | ---: | ---: | ---: |"]
    for row in all_rows:
        if row["cache"] == "steady":
            lines.append(f"| {row['M']} | {labels[row['method']]} | {row['tflops']:,.2f} | "
                         f"{row['peak_percent']:.2f}% | {row['median_ms']:.6f} | {row['round_cv_percent']:.2f}% |")
    lines += ["", "8192³ 的 Lt/默认吞吐比约 1.041×，但 Lt 的轮间 CV 为 4.87%，"
              "且五轮中有一轮排序反转，不能据此认定稳定的 4.1% 加速。"
              "12288³ 与 16384³ 上，本次调优配置的正式吞吐分别低于默认库约 0.49%、2.18%。", "",
              "候选选择基于短调优样本，正式结果来自独立、较长的测量。记录表明短样本"
              "选中的配置未必在后续运行条件下仍最快；没有将调优阶段的最快值当成最终峰值。", "",
              "## 频率与结果边界", "",
              "同一块 B300、148 SM、1100 W 功率上限，默认动态时钟。T3 的轮间 SM 频率快照"
              "覆盖约 1050–2032 MHz；因此不能假定全程运行于启动时的 2032 MHz。"
              "这些是轮间快照，不能确定每次 GEMM 的平均频率，也不能单独证明功率或温度限频。"
              "本次没有采集限频原因或 NCU，故不对大矩阵吞吐下降作单一原因归因。", ""]
    lines += ["", "## 验证、候选与环境", ""]
    valid = sum(len(c.get("validation", {}).get(m, [])) for c in result["cases"].values()
                for m in c.get("validation", {}))
    post = sum(row["post_passed"] for row in all_rows)
    candidate_records = [v for c in result["cases"].values() for v in c.get("candidates", {}).values()]
    candidate_valid = sum(v.get("validation", {}).get("passed", False) for v in candidate_records)
    lines += [f"正式方法三种子验证记录 {valid} 条，计时后通过 {post}/{len(all_rows)}；"
              f"候选 seed 0 全矩阵验证通过 {candidate_valid}/{len(candidate_records)}。", "",
              "原始逐次样本、全部候选参数、预热/校准次数、首次 AUTOTUNE 时间、实际载入库路径、"
              "GPU UUID、频率/功率快照保存在 [results.json](artifacts/results.json)。",
              "快照不代表全程锁频；本次不修改功率/时钟，不使用 CUDA Graph，不采集 NCU。", ""]
    lines += ["每个形状请求 32 个候选，库实际均返回 8 个，全部通过精度审计与 seed 0 校验；"
              "不是执行了 32 个候选，也不是从 32 个返回值中剔除了 24 个。"
              "这是当前描述符/偏好约束下的候选集，不代表库全部算法。", "",
              "| 形状 | 选中候选序号 | algo ID | tile ID | stage ID | cluster ID | Split-K | 所需 workspace |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, data in result["cases"].items():
        selection = data.get("selection")
        if selection:
            c = selection["config"]
            values = [str(c[key]["value"]) for key in ("algorithm_id", "tile_id", "stages_id", "cluster_shape_id", "split_k")]
            lines.append(f"| {name} | {data['selected_candidate']} | " + " | ".join(values)
                         + f" | {c['workspace_bytes']} B |")
    lines += ["", "以上为 API 返回的配置 ID，未将其推断成历史默认 cuBLAS 的 kernel 配置。"
              "所有选中候选均为 NONE 归约、Split-K=1，虽允许 64 MiB workspace，实际所需均为 0 B。", ""]
    for name, data in result["cases"].items():
        if len(set(data.get("sessions_used", []))) > 1:
            lines += [f"- **{name} 曾跨会话恢复**，各轮来源见 round_session_indices；不视作单会话配对证据。"]
    lines += ["", "## 数据来源", "",
              "- [完整原始结果](artifacts/results.json)",
              "- [汇总 CSV](summary.csv)",
              "- [原始样本与校验审计](audit.json)",
              f"- [历史对照]({old_path})",
              "- [NVIDIA 硬件规格](https://www.nvidia.com/en-us/data-center/hgx/)",
              "- [cuBLAS 13.1 文档](https://docs.nvidia.com/cuda/archive/13.1.0/cublas/index.html)", ""]
    (run / "RESULTS.md").write_text("\n".join(lines))
    if all_rows:
        with (run / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(all_rows)
    print(run / "RESULTS.md")


if __name__ == "__main__":
    main()
