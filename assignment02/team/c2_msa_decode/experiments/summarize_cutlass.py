"""Derive the fixed-pin external CUTLASS comparison report from raw data."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
WIDE = HERE / "results/cutlass-b300-20260913T020942Z"
SMALL = HERE / "results/cutlass-b300-20260913T022105Z"
PIN = "087c161814d4d9c735b46c21212a09e5f8eb92fa"
UP = f"https://github.com/vllm-project/MSA/blob/{PIN}/python/fmha_sm100"


def main():
    wide = json.loads((WIDE / "measurements.json").read_text())
    small = json.loads((SMALL / "measurements.json").read_text())
    controls = json.loads((SMALL / "probability-controls.json").read_text())
    combined = [dict(r,run="small",source=str(SMALL.relative_to(HERE))) for r in small]
    combined += [dict(r,run="wide",source=str(WIDE.relative_to(HERE))) for r in wide if r['batch'] > 16]
    combined.sort(key=lambda r:(r['tp'],r['batch']))
    out = ["# C2 固定上游 CUTLASS 对照与交叉点", "", "## 版本与独立运行范围", "",
           "此实验没有修改 vendored snapshot、CUTLASS decode planner/kernel/reduction 或生产 dispatch。只用独立 import shim 加载公开代码：未调用的 CuTe sparse-prefill adapter 替换成显式报错函数，避免完整 vLLM/CuTe-DSL 依赖。所有实际 decode CUDA 路径保持上游实现。", "",
           "vLLM 固定版本 `d4da0c55af3aa231b6209bf77871f3ed36eab0d2` 的 [fmha_sm100.cmake](https://github.com/vllm-project/vllm/blob/d4da0c55af3aa231b6209bf77871f3ed36eab0d2/cmake/external_projects/fmha_sm100.cmake) 指向 MSA `087c161814d4d9c735b46c21212a09e5f8eb92fa`；其 CUTLASS submodule 为 `eb61c911471867a5fd2466bfd8f29306cea6ebf8`。归档 SHA256：MSA `32efbae22ce41f85adf01dfd3ad98494a774591b10d00ea558a94d34170d579a`，CUTLASS `ffe392246cc3517017c4b91d14bbcb28aae28d26d1847b333371eb13bfec52eb`。", "",
           "环境与基线相同：B300、driver 580.95.05、CUDA toolkit 13.1.80、Torch 2.10.0+cu130、Triton 3.6.0。新增 apache-tvm-ffi 0.1.13.post3、ninja 1.13.0、pybind11 3.0.1。上游 JIT 同时生成 sm_100a/sm_103a/sm_100f，原始 build.ninja、生成 CUDA、.so、resource usage、SASS 均保留。", "",
           f"大 batch 原始任务：`{WIDE.relative_to(HERE)}`；完整小 batch 与概率控制：`{SMALL.relative_to(HERE)}`。第二次任务只复用了前一次固定公开源码生成的四个 .so，避免重复 GPU JIT。首次冷 plan 约 {wide[0]['cold_plan_seconds']:.3f} 秒，首个 split attention 编译约 {wide[0]['first_attention_seconds']:.3f} 秒，首次 nosplit 变体约 {wide[1]['first_attention_seconds']:.3f} 秒；它们不计入热 kernel 性能。", "",
           "## 比较口径", "",
           "两端共用同一随机页表/KV/索引和 seq8192、P128、K16、D128、GQA16、DQL1。Triton 使用 BF16 Q、FP8 KV、PDL=true；CUTLASS 额外将 Q 量化为 FP8，Q/K/V scale=0.25/0.25/0.5。每一表行的两端测于同一张 GPU、同一任务；小 batch 组与大 batch 组来自不同任务，不能据跨组绝对时间差推断微小效应。", "",
           "`attention` 包含上游计划选中的 forward 与可选 split-KV reduction；不含预先完成的 Q 量化、metadata 更新、冷 plan。`full` 是热 plan 下 Q 量化 + GPU metadata 更新 + attention 的 CUDA Graph：它不是服务端到端墙钟，不含 CPU seq_lens 拷贝、Python prepare 开销、冷 plan。这里 Q 量化用普通 PyTorch 多步算子而非生产可能使用的融合量化 kernel，故 full 只能描述本实验适配流程。各图 32 次调用，9 个样本，报每次调用中位微秒。", "",
           "额外提供 effective-Q Triton 与 FP64 gold：FP8 Q 反量化后用 BF16 表示。当前 power-of-two scale 使有效 FP8 Q/K/V 可被 BF16 精确表示，因此可以将额外 Q 量化误差与 attention 内部近似分开。数值 gate 没有因 CUTLASS 而放宽，本实验只记录有限性与误差，绝不当作候选 frozen 验收通过。", "",
           "B<16 是绕过静态 batch dispatch guard 后直接调用底层 wrapper 的受控实验；不表示原生产 dispatcher 支持或推荐这些形状。", "",
           "## 同卡交叉曲线", "", "speedup = 同行 Triton / CUTLASS；大于 1 表示 CUTLASS 更快。", "",
           "| TP | B | run | Triton µs | CUTLASS attention µs | CUTLASS full µs | attention speedup | full speedup |", "|---:|---:|---|---:|---:|---:|---:|---:|"]
    for r in combined:
        t=r['triton_original_q_graph']['median_us']; a=r['attention_graph']['median_us']; f=r['full_graph']['median_us']
        out.append(f"| {r['tp']} | {r['batch']} | {r['run']} | {t:.3f} | {a:.3f} | {f:.3f} | {t/a:.3f}× | {t/f:.3f}× |")
    out += ["", "在所测离散点上，TP1 的 attention-only 首个获益点是 B8，包含本实验 Q 量化/metadata 的 full 首个获益点是 B16。TP4 的 attention-only 首个获益点是 B32，full 首个获益点是 B64。这里没有测全部中间整数 batch，不能把这些离散点宣称为精确 crossover；TP4 没有复现上游注释所说的 B16。硬件、PDL、量化实现、缓存口径和数值约束都影响交叉点，静态 guard 是策略而非通用性能定律。", "",
            "## CUDA activity 与执行组织", "",
            "实际 trace 的 CUTLASS forward grid 恒为 [148,1,1]，block [384,1,1]，即 12 warps；168 registers/thread。trace 报 shared memory：split 180,992 B，nosplit 180,480 B。SASS 确认 `UTMALDG.4D`、`UTCQMMA` 和 FP8 probability pack。固定网格由 host-precomputed 工作表分配工作，不能把 148 CTA 等同于 148 个 SM 始终有有效工作。Torch activity 中 forward 的 estimated occupancy=0 是工具估计缺失/异常，未作为真实 occupancy 证据。", "",
            "上游 pack_factor=16，沿 GQA 组打包 query heads。wide 任务 TP1 B16 和 TP4 B16/32/64 选择 num_kv_splits=4，需要第二个 reduction；TP1 B32/B64 选择 nosplit=1，仅一个 forward。reduction block128、38 registers/thread、共享内存0；实际 grid TP1B16=[4,256,1]，TP4B16/B32/B64=[1,256/512/1024,1]。可见 CUTLASS 也不总是单 kernel。计划具体标量/张量大小和完整 trace 在逐形状目录。", "",
            f"固定源码 [`api.py:104–122`]({UP}/api.py#L104) 将可用 SM 数用于 CTA 计划并选择 GQA pack factor；[`api.py:553–670`]({UP}/api.py#L553) 构造工作表和 split 计划。不能用 Triton 的固定 256-CTA 分割公式推导此路径。", "",
            "## 额外误差来自哪里", "", "| TP | B | 对 effective FP8 Q 的 NRMSE | 对原 BF16 Q 的 NRMSE |", "|---:|---:|---:|---:|"]
    for r in combined:
        c=r['correctness']; out.append(f"| {r['tp']} | {r['batch']} | {c['against_effective_fp8_q']['nrmse']:.6f} | {c['against_original_bf16_q']['nrmse']:.6f} |")
    out += ["", f"源码证据：[`fmha_cutlass_sm100.cuh:61–64`]({UP}/csrc/include/fmha_cutlass_sm100.cuh#L61) 令 `Element=DTypeIn`，QK/PV accumulator 为 FP32；[`sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp:998–1032`]({UP}/csrc/include/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp#L998) 用 `NumericArrayConverter<Element,ElementQK,...>` 将 exp 后的 FP32 numerator 写为输入 Element。FP8 路径中 P 因此再舍入到 E4M3，而 Triton baseline 的 P dot operand 为 BF16。SASS 也包含 `F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C`，并非仅 KV 以 FP8 存储。", "",
            "为检验接入/布局/scale 是否造成约 2.6% 误差，增加人为重复单页控制：每个请求 16 个 top-k 条目都指同一逻辑页，所有 256-token KV tile 因而有相同 row maximum，避免复刻复杂的在线调度。独立参考对一页计算 FP64 logits 和 exp，再把未归一化 numerator 舍入 E4M3；分母仍用未舍入的 FP64 和，与上游 P 舍入口径对应。", ""]
    c=controls['repeated_page_random_q']
    out += [f"随机 Q 控制：CUTLASS 对数学 FP64 attention 的 NRMSE={c['against_mathematical']['nrmse']:.9f}；对独立 FP8-P 参考降至 {c['against_fp8_p_reference']['nrmse']:.9f}。Q=0 时 exp numerator=1，可被 FP8 精确表示，NRMSE={controls['repeated_page_zero_q']['nrmse']:.9f}，接近输出 BF16 舍入水平。源码、SASS 和这两个控制一致支持：P 侧 FP8 舍入解释了主要额外误差，不能只归因 Q 量化。该指纹不等于对所有边界的形式证明，也不替代 frozen accuracy suite。", "",
            "本比较保持初始性能输入和所有原始结果不变；没有为获得更漂亮的曲线而放宽误差阈值、修改量化 scale 或将 native FP8 attention 冒充同精度 BF16 算法。若将 CUTLASS 用作候选，必须先按相同 frozen gate 独立验收，当前外部对照的约 2.6% NRMSE 不能直接作为替代原算子的资格。", ""]
    (HERE / "CUTLASS_EXPERIMENTS.md").write_text("\n".join(out))
    (HERE / "cutlass_summary.json").write_text(json.dumps(dict(rows=combined,wide=wide,small=small,controls=controls),indent=2)+"\n")


if __name__ == '__main__':
    main()
