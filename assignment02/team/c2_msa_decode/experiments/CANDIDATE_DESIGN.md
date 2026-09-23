# C2 候选设计记录（baseline 实测之后）

记录日期：2026-09-13。候选实现从本记录之后开始。

## 已取得的输入证据

- `results/bench-b300-20260913T015939Z/measurements.json`：TP1/TP4 × B1/4/8/16 × BF16/FP8，共 16 个 baseline 配置；全部通过独立 FP64 常规长序列检查，NRMSE 约 0.00281–0.00298。
- `results/profile-b300-20260913T015957Z/baseline.csv`：TP4 B1 BF16 的 NCU detailed profile 成功。16 个 partial CTA，DRAM peak 约 1.85%，tensor elapsed 约 0.36%，没有填满设备；不能用显存带宽饱和解释该形状。
- 编译证据：partial 默认 4 warps、3 stages、73,732 bytes shared；实际主矩阵路径是 `mma.sync.m16n8k16`，没有 TMEM。
- 热地址 graph baseline 中 TP1 B16 BF16/FP8 为约 15.85/25.92 μs；FP8 更慢。FP8 是反量化后 BF16 dot，不是原生 FP8 dot。
- 独立验收方案已写入 `../validation/ACCEPTANCE.md`；候选不得用于生成或放宽其 baseline 冻结阈值。

## 选择的挑战路线：改良 Triton

优先检验三个有限、可归因的变化：

1. 将一个 128-token 物理页划为 32/64-token 的逻辑计算 tile。页表单位仍是 128，FP8 scale 仍按物理 token 索引。小 batch 可使用更多 split CTA；大 batch 可减少每 CTA 寄存器和 shared 需求。额外 partial/merge 成本可能抵消收益。
2. 分开扫描 `num_stages`、`num_warps` 与 split 数，避免默认 3 stages 对非常短循环占用过多 shared；实际效果由成对 graph 测量决定。
3. online softmax 内部维护 `(m, l, acc)`，以 `l_new = l_old*alpha + sum(p)` 更新分母，只在写出 partial LSE 时计算 log2。保存的 LSE 与 merge 数学口径保持不变，数值舍入重新验收。

不把融合为每 KV head 一个 CTA 作为首选：TP4 B1 会只剩一个 CTA，与测得的并行度不足方向相反。先保留两 kernel 结构和可选 PDL。

候选必须跳过完全处于 causal tail 之后的子页 tile，避免全 `-inf` logits 产生 NaN。空 split 写零和 `-inf` LSE；padding 行保持显式未定义/屏蔽契约。

## 实验步骤

1. 对固定长序列先扫描 tile/stage/warp/split，记录所有配置（包括负收益和编译失败），不只保存赢家。
2. 用独立 heldout seeds 及冻结的完整验收矩阵检验候选。
3. 在同一设备、相同输入、同一 PDL/Graph/分配口径下成对测量 baseline 与选择后的候选，交错顺序并报告原始样本。
4. 对选中候选补 NCU/编译资源解释；CUTLASS 比较另外记录 Q 量化、metadata 与支持域，不能偷换为相同输入精度。

这是实验假设和实现计划，不是性能结论。若没有稳健正收益，保留全部结果，并限定结论为本实现/本测量域。

## 第二阶段：依据 merge NCU 的独立变化

取得 `profile-matrix-b300-20260913T020433Z` 的 6 个额外代表 NCU 结果后，发现 merge 存在显著 shared-memory excess wavefront。例如 TP1 B1 S16 为实际 69,888 / 理想 8,448；TP1 B16 S4 为 577,536 / 86,016。

第一轮 TP4 B1 BF16 的 15 个 partial/stage 配置扫描全部正确，但未超过原 baseline，最快约 0.98×。这不是整个路线失败的证明；完整矩阵仍在执行。

下一项可隔离的实验是 feature-tiled merge：每个 CTA 只写同一 query/head 的 16/32/64/128 个输出通道，优先使用一个 warp，使 split 归约尽量留在 warp/寄存器中。先保持原 partial 完全不变，单独替换 merge，记录两 kernel chain 的变化，再决定是否与 subpage partial 组合。额外读取 LSE、CTA 数和布局代价均须实测。
