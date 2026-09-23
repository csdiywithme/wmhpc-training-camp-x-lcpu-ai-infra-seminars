"""Rebuild C2 baseline tables directly from saved measurements/NCU CSV."""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
HOT = RESULTS / "bench-b300-20260913T015939Z"
PDL = RESULTS / "bench-b300-20260913T020224Z"
NCU = RESULTS / "profile-matrix-b300-20260913T020433Z"
NCU_SMALL = RESULTS / "profile-b300-20260913T015957Z"


def ncu_rows(path):
    rows = list(csv.DictReader(path.open()))
    units = rows.pop(0)
    output = []
    for row in rows:
        output.append(dict(kernel=row["Kernel Name"], metrics={k:dict(value=v,unit=units[k])
                                                              for k,v in row.items() if k not in ("Kernel Name",)}))
    return output


def main():
    hot = json.loads((HOT / "measurements.json").read_text())
    pdl = json.loads((PDL / "measurements.json").read_text())
    ncu = {path.parent.name:ncu_rows(path) for path in sorted(NCU.glob("*/baseline.csv"))}
    ncu["tp4-b1-bf16"] = ncu_rows(NCU_SMALL / "baseline.csv")
    summary = dict(hot_source=str(HOT.relative_to(HERE)),pdl_source=str(PDL.relative_to(HERE)),
                   ncu_source=str(NCU.relative_to(HERE)),hot=hot,pdl=pdl,ncu=ncu)
    (HERE / "baseline_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    text = ["# C2 B300 原始基线测量与瓶颈证据", "", "2026-09-13 UTC 实测。此文件由 `summarize_baseline.py` 从原始 JSON/NCU CSV 重建。", "",
            "## 环境与口径", "",
            "GPU NVIDIA B300 SXM6 AC，148 SM、compute capability 10.3；driver 580.95.05，CUDA toolkit 13.1.80，Torch 2.10.0+cu130，Triton 3.6.0，NCU 2025.4.1.0。每次任务单卡，未锁定时钟；不同任务落在不同物理 GPU 上，不能将跨任务差值单独归因 PDL。", "",
            "输入固定 D=128、GQA=16、top-k=16、page=128、seq_len=8192、decode_query_len=1。top-k 张量采用 token-major backing 的 [Hkv,R,K] 转置视图；KV 为随机物理页池。FP8 为 E4M3FN 存储，K/V 标量 scale=0.25/0.5，Q 为 BF16。当前独立参考针对有效量化输入计算 FP64 QK/softmax/PV；这些正常形状检查不能代替完整边界与非 2 幂 scale 验收。", "",
            "`partial/merge/chain` 均先编译、预分配并捕获 CUDA Graph：每张 graph 连续 32 次调用，3 次 replay 预热，9 个 event 样本，表中为每次调用中位微秒。固定输入重复使缓存温热。`API` 为公开 Python wrapper，每次新输出及内部 workspace，40 次调用的 5 个 event 样本；包含 CPU 提交不足造成的 GPU 空闲，不能当作 kernel duration。", "",
            f"原始 graph/API 数据：`{HOT.relative_to(HERE)}`；PDL 开启：`{PDL.relative_to(HERE)}`。各目录保留运行脚本快照、环境、命令、逐形状结果、PTX/CUBIN/SASS、Triton IR 和编译元数据。", "",
            "## 完整小 batch 矩阵（PDL=false）", "",
            "| TP | B | KV dtype | S | partial CTA | merge CTA | partial µs | merge µs | chain µs | API µs | NRMSE |", "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in hot:
        t,b,s=r["tp"],r["batch"],r["splits"]
        text.append(f"| {t} | {b} | {r['dtype']} | {s} | {b*(4//t)*s} | {b*(64//t)} | {r['partial_graph']['median_us']:.3f} | {r['merge_graph']['median_us']:.3f} | {r['chain_graph']['median_us']:.3f} | {r['public_api']['device']['median_us']:.3f} | {r['correctness']['nrmse']:.6f} |")
    text += ["", "## PDL=true 的独立对照任务", "", "| TP | B | KV dtype | partial µs | merge µs | chain µs |", "|---:|---:|---|---:|---:|---:|"]
    for r in pdl:
        text.append(f"| {r['tp']} | {r['batch']} | {r['dtype']} | {r['partial_graph']['median_us']:.3f} | {r['merge_graph']['median_us']:.3f} | {r['chain_graph']['median_us']:.3f} |")
    text += ["", "部分 FP8 的 PDL chain 高于独立 partial+merge 之和，说明孤立 kernel 延迟相加不能重建有依赖 graph 的行为。此矩阵只能报告现象；同卡交错 PDL 开/关、graph 依赖时间线才足以隔离原因。性能候选必须与相同 PDL 设置的 baseline 在同一任务比较。", "",
             "## NCU 代表形状（PDL=false，cache-control=all）", "",
             "NCU detailed 对每个 kernel 22 次 replay，`cache-control=all` 每次清缓存；本表与上面的 hot graph 是不同口径。原始 .ncu-rep 和 CSV 均保存，未把以前 B300 profiling 失败的结果冒充计数器数据。此次 7 个代表输入均成功获取 partial/merge 计数器。", "",
             "| case | kernel | duration µs | DRAM read MB | DRAM read peak % | tensor elapsed % | active occupancy % | waves/SM | SMEM actual/ideal wavefront |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for case, rows in ncu.items():
        for r in rows:
            m=r['metrics']
            def v(key): return float(m[key]['value'].replace(',',''))
            # NCU chooses units per report, so convert explicitly.
            read=m['dram__bytes_read.sum']; mb=float(read['value'].replace(',',''))*{'byte':1e-6,'Kbyte':1e-3,'Mbyte':1,'Gbyte':1e3}[read['unit']]
            text.append(f"| {case} | {'partial' if r['kernel'].startswith('_gqa') else 'merge'} | {v('gpu__time_duration.sum'):.3f} | {mb:.6f} | {v('dram__bytes_read.sum.pct_of_peak_sustained_elapsed'):.3f} | {v('sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed'):.3f} | {v('sm__warps_active.avg.pct_of_peak_sustained_active'):.3f} | {v('launch__waves_per_multiprocessor'):.2f} | {v('memory_l1_wavefronts_shared'):.0f}/{v('memory_l1_wavefronts_shared_ideal'):.0f} |")
    text += ["", "## 编译资源和指令", "", "| TP | B | KV dtype | partial regs | Triton n_spills | dynamic SMEM bytes | merge regs | merge dynamic SMEM bytes |", "|---:|---:|---|---:|---:|---:|---:|---:|"]
    for r in hot:
        a,b=r['compiled']
        text.append(f"| {r['tp']} | {r['batch']} | {r['dtype']} | {a['n_regs']} | {a['n_spills']} | {a['metadata']['shared']} | {b['n_regs']} | {b['metadata']['shared']} |")
    text += ["", "所有 kernel 为 4 warps；partial TMEM=0，实际 SASS 使用 `HMMA.16816.F32.BF16`。FP8 版本静态 SASS 还包含 128 条 E4M3 unpack、256 条 F16→BF16 转换、128 条 FMUL2；这些是静态指令计数，不是动态执行占比。FP8 KV 会先转 BF16、做 FP32 scale、再 round BF16，因此不能套用 native FP8 Tensor Core 峰值。partial 动态 SMEM=73,732 B，NCU 另计 driver 1,024 B 和分配粒度；通常最多 3 个 CTA/SM，211/195 regs 版本由寄存器限制至 2 个 CTA/SM。BF16 和 B1 FP8 的 Triton `n_spills` 字段为 0；B≥4 FP8 partial 的字段为 2，不能把这个字段直接解释成流量字节数。NCU 对两个 B16 FP8 代表点均记录 2,048 个 local spilling requests（local load/store 各 1,024），且 SASS 有 STL/LDL，与编译信息一致；B1 FP8 及 BF16 代表点对应请求为 0。编译器会按 shape/stride 特化，不能用一个 B1 的寄存器数代表整个矩阵。", "",
             "## 可支持的结论与边界", "",
             "1. TP4 B1 只有 16 个 partial CTA，远少于 148 SM；NCU DRAM read 仅峰值约 1.85%，tensor elapsed 约 0.36%。小 batch 首先表现为 grid 不足和固定延迟，不能笼统称为 HBM 带宽已饱和。TP1 B16 BF16 的 DRAM read 已升至峰值约 44%，瓶颈随 batch 改变。", "",
             "2. FP8 read 约为 BF16 一半，但端到端更慢。TP1 B16 partial 的 SMEM wavefront 为 2,824,704，理想值 2,005,504；BF16 为 1,546,752，理想值相同。额外转换、布局交换和 bank conflict 是有证据的候选解释。B1 FP8 的寄存器占用也更高；B16 的寄存器趋势反转，所以不能把所有 FP8 退化归因寄存器。", "",
             "3. merge 的 SMEM wavefront 明显超出理想值：S16 的 TP1 B1 为 69,888/8,448，S4 的 TP1 B16 为 577,536/86,016。支持检验 merge 布局/归约组织与 launch 数量。按独立 merge 时间估算，TP1 B1 BF16 merge/chain 约 34.8%，TP1 B16 FP8 约 15.2%；即使完全消除 merge，也只能给出约 1.53×/1.18× 的粗略 Amdahl 上界（孤立 kernel 时间和 chain 并不严格相加，故不是可兑现的精确上界）。它不证明任意特定实现必然更快；候选仍须同卡测量与完整数值验收。", "",
             "4. NCU 的 PC sampling 数量较小（TP4 B1 partial 总量仅几十），long/short scoreboard 的绝对计数只能作线索，不应包装成稳定百分比或精确瓶颈占比。active occupancy 与 elapsed tensor utilization 的分母不同，不能相互直接相减。", "",
             "5. `dram__bytes_write.sum=0` 仅表示该次计数窗口未观察到 HBM 写回；partial/out 的 store 仍存在，可能留在 L2。不能用它删掉理论 workspace 流量。SMEM wavefront 是 bank-conflict 工作量指标，不等同于独立的算法字节数。", "",
             "6. 本数据是固定活跃页的 hot graph 加独立 cold NCU。未覆盖连续新请求/冷热混合、页表不命中、生产调度与服务尾延迟。首个单形状与完整矩阵的 TP4 B1 BF16 chain 分别约 5.593/6.050 µs，说明跨任务机器与时钟差异不可忽略；候选 speedup 必须使用同一任务的 baseline。", ""]
    (HERE / "BASELINE_PROFILE.md").write_text("\n".join(text))


if __name__ == '__main__':
    main()
