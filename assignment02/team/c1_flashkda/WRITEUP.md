# C1：FlashKDA 为什么仍使用 SM80 MMA？

实验日期：2026-09-13。目标平台：单张 NVIDIA B300 SXM6 AC。本文对应 [TASK.md](TASK.md) 的复现、六个讨论点和挑战实现；答辩材料见 [DEFENSE.md](DEFENSE.md)。按用户确认，课堂环节以答辩材料和 [内部独立审查记录](analysis/FINAL_REVIEW.md) 交付；未实际举行课堂讲述或答辩。

## 1. 结论与提交内容

**本次证据支持保留 C16 / SM80 MMA 路径作为默认实现，并继续研究有条件启用的 SM100 后端。** 原版在 H64 / H96、同门函数的 FLA 对照下获得 1.48–2.89× 加速；实际 B300 二进制使用传统 `HMMA`，没有自动变成 `tcgen05`。K2 的主要问题是有限的独立状态链、链内依赖和片上数据流，不能仅用全卡 Tensor Core 峰值解释。

挑战选择了任务允许的“并行度重构”：把同一 head 的 value columns 分给两个 CTA。该版本通过逐位正确性检查，但所有六个正式计时形状均变慢，加速比仅 0.571–0.751×。这否定了本次具体改法，不构成对所有 SM100 设计的否定。

| 交付项 | 代码 / 文档 | 可核验证据 |
|---|---|---|
| 固定版本构建与实验入口 | [modal_experiments.py](modal_experiments.py)、[run_experiments.py](run_experiments.py) | 每次运行的 `commands.json`、版本和退出码 |
| 并行度挑战 | [challenge_build.py](challenge_build.py) | 独立 `flash_kda_split2` 模块、生成的 patch、正确性与性能数据 |
| 指数与求逆机制 | [chunk_numeric_gpu.py](chunk_numeric_gpu.py) | 6 个范围案例、45 个逆矩阵案例、实际 MMA SASS |
| SM80 / tcgen05 tile 比较 | [tile_microbench.cu](tile_microbench.cu) | 64 个配置通过全部 CTA 检查，320 个计时样本与 SASS |
| 详细推导 | [C1_THEORETICAL_ANALYSIS.md](analysis/C1_THEORETICAL_ANALYSIS.md) | CPU 恒等式、成本和数值反例 |
| GPU 实验记录 | [C1_EXPERIMENTS.md](analysis/C1_EXPERIMENTS.md)、[C1_NUMERIC_EXPERIMENTS.md](analysis/C1_NUMERIC_EXPERIMENTS.md) | 原始数据与测量边界 |

上游 `FlashKDA/` 与 `fla_kda_ref/` 快照保持不变。挑战脚本在远端复制树中生成实现；实验代码和报告位于题目目录之外层。理论报告保留写作时的历史状态，本文的 GPU 结论采用后续实际运行记录。

## 2. 环境、语义与原版复现

版本固定为 FlashKDA `1ce47ea`、CUTLASS `5c149f5`、FLA `a3edffc39eb5a3d45e9deab5ff9ec4f14f88474d`。环境为 B300 CC 10.3、148 SM、驱动 580.95.05、CUDA toolkit 13.1.80、Torch 2.10.0+cu130、Triton 3.6.0、Nsight Compute 2025.4.1；本次只构建 `sm_103a` cubin。`FLA_FLASH_KDA=0` 保证 FLA 对照不会再 dispatch 到 FlashKDA。

必须先对齐算子语义。FlashKDA 接收 raw gate 和 beta logits，内部使用

\[
g=-5\,\sigma(e^{A_{log}}(g_{raw}+dt\_bias)),\quad \beta=\sigma(\beta_{raw}).
\]

当前固定版本 FLA 需要显式 `safe_gate=True` 才选择这一 bounded gate。原 benchmark 可运行，但未传该参数，因此先保存原样结果，再生成仅补充 `safe_gate=True` 与现行 `state_v_first=True` 参数名的派生脚本。报告中的同语义加速比使用后者。状态接口是 value-first `[N,H,V,K]`；naive 数学参考使用 `[N,H,K,V]`，D=128 时转置错误不会改变 shape，必须显式处理。

H96、D128、总 T8192；按官方设置 30 次 warmup、200 iterations × 5 repeats，单位 ms：

| 序列组织 | FlashKDA，BF16 state | FLA chunk，同 bounded gate | FLA / FlashKDA |
|---|---:|---:|---:|
| 1×8192 | 1.0789 | 2.0788 | 1.93× |
| 1300 / 547 / 2048 / 963 / 271 / 3063 | 0.8892 | 2.1029 | 2.36× |
| 8×1024 | 0.7110 | 2.0575 | 2.89× |

原始数据及派生 benchmark：[official 结果目录](results/c1-official-h96-20260913T020630921340Z/)。原脚本的 no-state、FP32-state、FLA GDN 列也保存，但 GDN 是另一种 gate 语义，不计作同算子加速。FP32-state 列只改变状态接口精度，内部状态仍为 BF16。官方初始状态使用很大的 `arange` 值，适合复现原表；精度实验另用随机状态。

H64 附加形状也按相同设置完成原样与同门函数复测，见 [H64 official 结果目录](results/c1-official-h64-20260913T022812279519Z/)。同门函数结果如下，单位 ms：

| 序列组织 | FlashKDA，BF16 state | FLA chunk，同 bounded gate | FLA / FlashKDA |
|---|---:|---:|---:|
| 1×8192 | 0.9882 | 1.4672 | 1.48× |
| 不等长 6 序列 | 0.6704 | 1.4839 | 2.21× |
| 8×1024 | 0.4826 | 1.3869 | 2.87× |

本文不将本次 B300 与仓库 GB200 表的差异解释为架构因果收益。

## 3. 统一数学模型与实现分工

对一个 head，数学状态为 S∈R^(Dk×Dv)，令 D_t=diag(exp g_t)：

\[
\bar S_t=D_tS_{t-1},\quad u_t=\beta_t(v_t-\bar S_t^Tk_t),\quad
S_t=\bar S_t+k_tu_t^T,\quad o_t=S_t^Tq_t.
\]

下文将输出 scale 吸收进 q。chunk 内定义 G_i=Σ_(t≤i)g_t，K_d[i]=k_i⊙exp G_i，K_inv[i]=k_i⊙exp(−G_i)，Q_d 同理，K_r[i]=k_i⊙exp(G_C−G_i)。令

\[
L=\operatorname{strictLower}(\operatorname{diag}(\beta)K_dK_{inv}^T),\quad
R=(I+L)^{-1},\quad M=\operatorname{lower}(Q_dK_{inv}^T),
\]

则

\[
U=R\operatorname{diag}(\beta)(V-K_dS_0),\quad
O=Q_dS_0+MU,\quad
S_C=\operatorname{diag}(e^{G_C})S_0+K_r^TU.
\]

K1 不依赖入口状态，负责 q/k 归一化、gate / beta 激活、chunk 内矩阵与逆；K2 得到入口状态后执行上述更新，再进入下一 chunk。源代码入口分别为 [fwd_kernel1.cuh](FlashKDA/csrc/smxx/fwd_kernel1.cuh)、[fwd_kernel2.cuh](FlashKDA/csrc/smxx/fwd_kernel2.cuh)，求逆见 [utils.cuh](FlashKDA/csrc/smxx/utils.cuh) 的 `neumann_inv_fused_1warp`。

## 4. 讨论点 1：C16 的三个理由与 C32 / C64 的破坏方式

三个约束共同决定设计，不存在对全部输入一致的“先坏哪一个”。

| 项目 | C16 | C32 | C64 |
|---|---:|---:|---:|
| 最坏独立指数 exp(±5C) | exp(±80)，有限 | exp(±160)，溢出 | exp(±320)，溢出 |
| dense Neumann GEMM 数 | 6 | 8 | 10 |
| 逆矩阵计算 FLOP / token / head | 3,072 | 16,384 | 81,920 |
| 总主 GEMM FLOP / token / head，D128 | 117,760 | 147,456 | 245,760 |
| 原组织的 shared memory 纸面预算 | 约 96 KiB | 约 158 KiB | 约 306 KiB |

最后两列是假设沿用当前 dense doubling 和 buffer 结构的外推，不是已经实现的 C32 / C64 KDA。实际 C16 K2 分配 98,432 B；C64 的纸面分配超过单 CTA 容量，需要重构 stage / 布局。

**指数范围。** BF16 与 FP32 的指数位数相同；当前 `ex2.approx.ftz.f32` 先以 FP32 求值并清除 subnormal，再转 BF16。保证独立因子处于正常范围的保守条件约为 `5C < 126 ln 2 = 87.3365`，因此 C16 自然匹配，17 也满足纸面界。GPU 在 g=−5 时测得第 18 个 token 的 exp 归零、逆因子 Inf，C32 / C64 恢复项出现 `0×Inf`。直接算 exp(G_C−G_i) 可避免这个 NaN，但远距离项仍可能下溢。中心化使 C32 的跨度降为约 ±80；它只改善因子范围，完整 rescale 还必须补偿状态坐标、入口和末状态，C64 的 ±160 仍不够。

**有限级数不等于稳定算法。** 严格下三角 L 满足 L^C=0，因此 R=I−L+L²−… 是有限恒等式，不要求 ||L||<1。dense doubling 的代价为 `4(log2 C−1)C³` FLOP。对齐单位 keys、β=1、g=0 时，L 为严格下三角全 1；精确逆只有主对角 1 和第一下对角 −1，但 `(L^p)_ij=binom(i−j−1,p−1)`。C32 / C64 的部分和 P₈=I−L+L²−…−L⁷ 已超过 FP16 范围；这里 P₈ 与幂 L⁸ 是两个不同阶段。

GPU 机制实验的关键结果如下，详见 [数值报告](analysis/C1_NUMERIC_EXPERIMENTS.md)：

- C16、β=0.990234375：显式 F16 MMA 求逆最大绝对误差 1，角点 −1，而解析参考约 −7.10e−29；FP64 三角求解本身可能把这样的小角点算为零，不能把解析非零值冒充 FP64 求解实测值。没有 Inf 也可发生严重抵消。
- C32 / C64、β=1：首个 Inf 均出现在 P₈，最终逆矩阵全部 NaN。
- 全 FP32 doubling 同样不足：C32、β=1 最大误差 8，C64 最大误差约 2.15e9；后者 κ∞(I+L)=128，不能将结果归因于极端病态的线性系统。
- 固定种子下三个随机归一化 keys 案例的半精度最大误差仅约 5.28e−5 至 1.06e−4，说明常规随机测试可掩盖结构性反例。

**形状匹配。** C16 的 16×16 求逆天然适配一个 warp 的两条 m16n8k16 atom，寄存器内串联不需要 TMEM 往返。大 chunk 减少边界次数，却增加求逆、C²D 项和资源压力；不能用 chunk 数减少倍数估算加速。

## 5. 讨论点 2：tcgen05 能否匹配 C16？

普通 dense、非 `.ws`、单 CTA 的 FP16/BF16 `tcgen05.mma` 支持 M64 / M128，N 为 8–256 范围内 8 的倍数，单指令 K16。见 [PTX 形状规范](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-matrix-shape)。**K1 的 16×16 逆有 padding 问题，K2 并非都只有 25% 形状利用率。**

| 计算 | 原逻辑 M×N×K | 用 (AB)^T=B^TA^T 后 | 普通 tcgen05 形状利用率 |
|---|---|---|---:|
| K1 Gram / 一次逆 GEMM | 16×16×128 / 16×16×16 | 输出仍 16×16 | 孤立 pad M64 时 25% |
| K2 K_dS / Q_dS | 16×128×128 | 128×16×128 | 100% |
| K2 R×residual / M×U | 16×128×16 | 128×16×16 | 100% |
| K2 state update | 128×128×16 | 不需改变 | 100% |

利用率只表示有效乘积占 padded 算术量的比例，不是峰值利用率。实际还要处理 TMEM 累加器、异步完成、beta / gate 的标量操作、U 的布局、BF16 写回与下一阶段操作数。当前 K2 的寄存器累加和转置融合较紧，不能只替换 MMA 调用而忽略这些成本。

最终可信 [tile 结果目录](results/c1-tile-h96-20260913T022453478641Z/) 包含源码快照、hash、SASS 与 [汇总](results/c1-tile-h96-20260913T022453478641Z/tile-summary.json)。两种实现 × 四种形状 × 四种 CTA 数 × 两种内部重复次数，共 64 个配置，全部 CTA 的显式有限值与 CPU 参考检查通过，最大误差 0；每配置保存 5 个计时样本，共 320 个样本。实测源码 SHA-256 为 `878273ce2a4f5b4a4129374a49b45833f592daf2eb9e005557061c9057367e41`。

实验比较固定 shared 输入下的 FP32 依赖累加链，最后写回一次输出。tcgen05 每轮 GEMM 后 commit / wait，由 issuer warp 独自推进 phase，循环结束后 CTA 同步再读取 TMEM。以包含 20 次 kernel 的 graph 重放 20 次计时，每个样本除以 400。下表是 96 CTA 下 `(mean(t64)−mean(t1))/63` 的增量，单位 ns：

| M×N×K | SM80 | tcgen05 | SM80 / tcgen05 |
|---|---:|---:|---:|
| 128×16×128 | 182.55 | 242.20 | 0.75× |
| 128×16×16 | 23.25 | 83.48 | 0.28× |
| 128×128×16 | 193.59 | 111.83 | 1.73× |
| 64×16×16 | 11.40 | 73.49 | 0.16× |

这支持“瘦形状仅换指令不保证更快”，并在 state update 形状发现可继续研究的增量收益；尚不能宣布完整 K2 获益。state update 完整单轮 kernel 为 SM80 8.632 / tcgen05 13.620 μs，64 轮为 20.828 / 20.665 μs，说明固定成本和摊销程度会改变结果。最后一行两侧均计算完整 M64，不是用原始 M16 inverse 与 padded 版本作公平端到端比较。

SASS 显示 SM80 操作数可被编译器保留在寄存器，因而不能宣称每轮都重新读取 shared。增量表示固定 grid 增加一整轮工作的边际时间，包含同步和数据依赖，不是单条 MMA 延迟或每 CTA 延迟。它也没有比较新指令的满流水峰值。

独立审查修正了旧版本的两处方法问题：全 CTA 独立 parity wait 可能被 issuer warp 越过 phase；只检查第 0 CTA 且 `fmaxf` 可掩盖 NaN。最终版本分别改为 issuer-only phase 推进与全 CTA 显式有限值检查，并重新运行。旧目录保留追溯，不用于本文最终性能结论。

## 6. 讨论点 3：并行度从哪里来，挑战为什么负收益？

当前 K2 的 CTA 数为 `N×H_local`，每 CTA 遍历一个序列一个 head；H_local=96/TP。T8192、C16 时，1 条序列是一条 512-chunk 链。H96 仅 96 CTA，H12（TP8）仅 12 CTA，均小于 B300 的 148 SM。增加 T 只延长链，不会增加链数。

| 候选 | 理由 | 反例 / 代价 |
|---|---|---|
| 增加真实独立序列 | 增加独立状态链 | 不能把同一长序列伪装成零状态短序列 |
| value-column split | 不同状态列独立，无归约 | 重读只读输入、重复 shared allocation、store 与寄存器代价 |
| 多 head / CTA | 交错独立工作可掩盖局部等待 | 减少 CTA 数，不能解决低 NH 的 SM 覆盖不足；朴素拼接有跨 head 交叉项 |
| persistent / 队列 | 改善变长负载平衡 | 当前 K2 已驻留整条链，队列不能创造独立链 |
| 两 CTA / head | 重新组织资源与数据路径 | cluster 同步、瘦 tile 和协作成本需要实测 |
| 时间 affine scan | 变换复合有结合性 | 一般转移矩阵变稠密或增秩，O(D³) 复合及中间存储可能抵消收益 |

本次实现 value split2。按列分 S 和 V 后，每列的 U、输出、后继 state 均只依赖同一列；源码原有 4 个 compute warp 各处理 32 个 value columns，候选将其分到两个 CTA，各保留 2 个 compute warp。K1、每列算术顺序与 BF16 舍入点不变；输入 TMA 与完整 shared state 刻意保留，store warp 改为只写自己负责的范围。

30 组小形状 precision 案例的候选 output / state 与 baseline 全部逐位一致；8 个 state 有无 × BF16 / FP32 接口组合通过；另有 H96 三个完整 T8192 形状逐位通过。因 baseline 同时对官方舍入参考及独立 FLA naive 检查，候选具有可追溯的正确性链。数据见 [challenge](results/c1-challenge-h96-20260913T020956599379Z/) 和 [large-shape correctness](results/c1-challenge_profile-h96-20260913T021340887396Z/large-shape-correctness.json)。

20 warmup、100 iterations × 5 repeats，单位 ms；加速比定义为 baseline / split2：

| H | 序列 | baseline | split2 | 加速比 |
|---:|---|---:|---:|---:|
| 12 | 1×8192 | 0.8370 | 1.4058 | 0.595× |
| 12 | 不等长 6 序列 | 0.3568 | 0.4751 | 0.751× |
| 12 | 8×1024 | 0.1604 | 0.2293 | 0.699× |
| 96 | 1×8192 | 1.0761 | 1.8648 | 0.577× |
| 96 | 不等长 6 序列 | 0.8861 | 1.4493 | 0.611× |
| 96 | 8×1024 | 0.7092 | 1.2423 | 0.571× |

候选 H96 K2 的 grid 从 96 增至 192，NCU 时间却从 793.440 增至 1,583.040 μs；shared 仍为 98,432 B/CTA，寄存器从 65 增至 80，新增约 157 万 / 118 万 local load / store 请求，baseline 为零。DRAM 读从 0.886 增至 1.106 GB，tensor elapsed 比例从 18.27% 降至 9.16%。增加 CTA 的收益未覆盖这些代价。没有做逐项消融，因此不能把全部损失单独归因于 spill、重复读或 store 中任何一个因素。

## 7. 讨论点 4：瓶颈必须按 kernel 与并行域回答

SASS 来自实际安装的 `sm_103a` 扩展，包含 `HMMA.16816.F32.BF16` 与 `HMMA.16816.F16`，未出现 HGMMA / UTCHMMA。静态计数涵盖多个模板实例，不当作单次调用的动态指令数。

H96、1×8192 的 NCU 结果：[profile 目录](results/c1-profile-h96-20260913T020442527901Z/)。

| 指标 | K1 | K2 |
|---|---:|---:|
| duration | 272.544 μs | 793.440 μs |
| grid / threads | 49,152 / 256 | 96 / 192 |
| dynamic shared / CTA | 21,248 B | 98,432 B |
| registers / thread | 32 | 65（分配按 72 计） |
| waves / SM | 41.51 | 0.32 |
| DRAM 总吞吐 | 4.529 TB/s | 1.353 TB/s |
| L2 throughput，elapsed peak | 72.32% | 25.55% |
| tensor active，elapsed / active | 5.64% / 5.73% | 18.27% / 28.40% |
| active warps，active peak | 96.24% | 9.37% |

K1 有高并发与较高内存 / L2 活动，同时承担门函数、归约和小矩阵工作，证据偏向访存及准备阶段成本。K2 的全卡 HBM 与 tensor 都不饱和，不能称为全卡 compute peak 或 DRAM 带宽饱和；低 grid 与状态链临界路径更能解释结果。

交叉验证尤其有力：[H12 profile](results/c1-profile-h12-20260913T020932999579Z/) 中 K1 降至 41.728 μs，K2 却仍为 786.528 μs；K2 tensor elapsed 只有 2.24%，active 仍约 28.60%。减少 8 倍 head 几乎没有缩短单条链的时间，说明平均吞吐不足包含“可并发工作不足”，不能简单归咎于某种 Tensor Core 指令。

纸面主 GEMM 强度约 38.6 FLOP / 请求字节，但这不是实测 HBM 强度。每 C16 / head 的 workspace 为 13,824 B；状态留在 shared memory，不能把它按每 chunk 都往返 HBM 计流量。正式判断联合使用 `gpu__time_duration.sum`、DRAM bytes / throughput、L2、tensor elapsed / active、grid / waves / eligible warps、shared / registers / local requests，以及 short-scoreboard / wait 等采样。stall 名称只作定位线索，不能直接相加成时间份额。

NCU 使用 detailed、22 passes、cache flush、未锁频，与普通 timing 不同；上述 duration 配合 counter 分析，正式速度比使用独立 benchmark 样本。assignment 4.5 的瘦 GEMM 是形状与 roofline 参照，投影 GEMM 和带递推状态的 K2 不是相同数据流。

## 8. 讨论点 5：BF16 状态精度怎样验证？

外部 FP32 state 不代表内部 FP32 持久状态。当前每 chunk 更新后仍转 BF16，因此采用三层检查：

1. 官方 `tests/torch_ref.py` 模拟相同近似和舍入，检查实现是否忠实于该数值路径。
2. 固定 `fla_kda_ref/naive.py::naive_recurrent_kda` 独立逐 token 递推；它内部显式 FP32，不能称为 FP64 gold。正确处理 raw activation、q/k 归一化、状态转置和 varlen。
3. 禁用 FlashKDA dispatch 的 FLA Triton chunk，同样启用 bounded gate，与同一 naive 参考比较。

已完成 H4、两个 seed、长度 16 / 17 / 97 / 1024 与 varlen 17 / 33 / 65，随机 / 弱 / 强衰减共 30 组。output / state 均有限，baseline 与官方舍入参考全部逐位一致。对 naive 的最大相对 RMSE，按 gate 类别聚合如下：

| gate | FlashKDA output | FlashKDA state | FLA chunk output | FLA chunk state |
|---|---:|---:|---:|---:|
| 随机 | 0.5162% | 0.4877% | 0.3385% | 0.2291% |
| 弱衰减 | 0.7213% | 0.7467% | 0.4450% | 0.4190% |
| 强衰减 | 0.5006% | 0.4896% | 0.3099% | 0.1819% |

数据见 [precision.json](results/c1-challenge-h96-20260913T020956599379Z/precision.json)。另外完成 H4、两个 seed 的 T8192 / T32768 弱衰减长序列，见 [long precision](results/c1-long_precision-h4-20260913T022617825299Z/precision.json)。相对同一 FP32 naive 参考的 RMSE 如下：

| seed / 长度 | FlashKDA output | FlashKDA state | FLA chunk output | FLA chunk state |
|---|---:|---:|---:|---:|
| 0 / 8192 | 0.7459% | 0.7026% | 0.4772% | 0.4076% |
| 0 / 32768 | 0.8166% | 0.7505% | 0.4838% | 0.4079% |
| 1 / 8192 | 0.7471% | 0.7029% | 0.4873% | 0.4173% |
| 1 / 32768 | 0.7525% | 0.7057% | 0.4634% | 0.3924% |

这四组 baseline output / state 同样与官方舍入 oracle 逐位一致，独立参考比较结果均有限。按 1024 token 分窗，FlashKDA output 相对 RMSE 最差窗口为 0.8238%。不同长度的随机生成不保证共享相同前缀，不能把 T8192 与 T32768 两行解释为同一序列的严格单调漂移实验；窗口指标也属于整实现误差。

这个误差包含整个实现的输入舍入、gate 近似、求逆、U 和 state，不能全归因于 BF16 state。若要独立测状态贡献，必须固定其他算术边界，仅改变持久状态存储；本次未完成这种完整 kernel 消融，也未跑真实 checkpoint 的困惑度 / 长上下文任务。

理论反例说明验证域不能只用随机数据：令 key=e₁、query=e₂、v=0，S₀=e₂e₂ᵀ，第二通道激活后 gate=−1e−4。该方向不被 delta 更新覆盖；C16 的精确衰减 exp(−0.0016)=0.998401，舍入 BF16 回到 1，连续保存会停滞。T65536 的真值约 0.001425。这是 [CPU 理论检查](analysis/theory_checks.py) 的隔离反例，不是本次完整 CUDA kernel 的 GPU 数据。

精确归一化下，A_t=(I−βkkᵀ)D_t 非扩张；但 gate 可接近零，不能给全部输入统一严格收缩率。非扩张不能推出舍入误差可忽略。发布判断仍需真实分布、弱衰减与低维 key、长序列、误差随位置变化以及状态消融。

## 9. 讨论点 6：作者应否发布 SM100 专版？

我们的选择是：**保留现有路径；SM100 作为有精度和性能门槛的专用后端研发，不发布当前 split2。**

保留理由是实际端到端收益、C16 的范围与形状契合、紧密寄存器融合，以及新实现尚未证明整体收益。这里的可移植性是当前项目支持的架构之间复用实现；源码还使用 TMA 等新特性，“使用 SM80 MMA”不等于能在 Ampere 上运行。构建脚本列出 90a / 100a / 103a / 120a，本次仅验证 103a，不能替其他架构作实测保证。

继续研发的理由是 K2 经转置后形状合法，TP8 的状态链并行不足有明确证据，state update 的依赖链微基准也出现 1.73× 增量收益。专版必须同时优化输入 / state 布局、TMEM 消费、标量更新、缓冲容量和跨阶段融合；微基准的有利点尚未成为完整 K2 的端到端收益。

建议下一步优先消融 value split 的全状态分配、输入复制和 store 路径，并在 H12 / H96、固定 / varlen、BF16 / FP32 接口上保留精度基线。另一条路线是保留 K1，只重构 K2；大 chunk 则先解决指数表示和稳定三角求解，再谈 tile。任何发布条件都必须包括完整 fwd 的持续正收益与同语义精度，而不是孤立 microbench 的有利点。

## 10. 复现命令与已知局限

在仓库根目录，配置可运行 B300 的 Modal 后执行。每次实验生成新的时间戳目录，不覆盖旧证据：

```sh
c1_run() {
  uv --cache-dir /private/tmp/a02-uv-cache run --offline --no-project \
    --with modal==1.5.5 python -m modal run \
    assignment02/team/c1_flashkda/modal_experiments.py "$@"
}
c1_run --mode official --heads 96
c1_run --mode official --heads 64
c1_run --mode profile --heads 96
c1_run --mode profile --heads 12
c1_run --mode challenge --heads 96
c1_run --mode challenge_profile --heads 96
c1_run --mode precision --heads 4
c1_run --mode long_precision --heads 4
c1_run --mode numeric --heads 96
c1_run --mode tile --heads 96
```

上述 `--offline` 仅限制本地 uv 依赖解析；远端镜像仍按固定 commit 安装公开源码和依赖。若本地无相同缓存，可安装 Modal 1.5.5 后直接使用同一 `python -m modal run` 入口。已缓存镜像会复用构建层，计时不含安装 / JIT。单 function 有明确超时、无自动重试，失败会保留退出码与原始输出。

具备 CUDA 编译器和上述 Torch 环境的 B300 上，数值机制脚本可独立执行；CPU 理论脚本只依赖 Python 标准库：

```sh
python assignment02/team/c1_flashkda/chunk_numeric_gpu.py --output /tmp/c1-numeric-new.json
python assignment02/team/c1_flashkda/analysis/theory_checks.py
```

数值脚本当前工作版本与原实测快照有一次 causal-pair 诊断舍入差异，复刻原始 JSON 应使用结果目录内 hash 一致的脚本，详见数值报告。CPU 理论检查仅将 JSON 输出到标准输出，不修改既有结果。

本次局限：单 B300、固定软件版本与有限种子；普通计时未锁频；candidate 与 baseline 顺序测量，未完成跨卡交错复验；没有完整 C32 / C64 KDA、完整 tcgen05 K2、模型级质量评估或真正 FP32 持久状态消融。旧 tile 数据不作为最终结论。课堂环节按已确认范围交付答辩材料与内部独立审查记录，未举行实际课堂活动。
