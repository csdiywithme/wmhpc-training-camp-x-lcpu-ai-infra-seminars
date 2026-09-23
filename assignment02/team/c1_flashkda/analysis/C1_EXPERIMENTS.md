# C1 GPU 实验记录（B300，2026-09-13）

本文件记录实际执行的实验，理论推导见 `C1_THEORETICAL_ANALYSIS.md`。上游 `FlashKDA/` 与 `fla_kda_ref/` 快照不作修改；挑战实现由 `challenge_build.py` 在远端副本生成独立 Python/CUDA 模块。

## 1. 可复现实验环境

- Modal 单张 NVIDIA B300 SXM6 AC，计算能力 10.3，148 SM，驱动 580.95.05。
- CUDA toolkit 13.1.80；PyTorch 2.10.0+cu130；Triton 3.6.0；Nsight Compute 2025.4.1。
- FlashKDA commit `1ce47ea`、CUTLASS 子模块 `5c149f5`，仅生成 `sm_103a` cubin。
- FLA 完整 clone 固定 `a3edffc39eb5a3d45e9deab5ff9ec4f14f88474d`，`FLA_FLASH_KDA=0` 强制 Triton 对照，避免对照组反向 dispatch 到 FlashKDA。
- 每次单 GPU function 最长 900 秒，无重试；原始命令、退出码、标准输出、环境和产物保存在带 UTC 时间戳的 `results/c1-*` 目录。

本地命令前缀：

```sh
uv --cache-dir /private/tmp/a02-uv-cache run --offline --no-project --with modal==1.5.5 python -m modal run assignment02/team/c1_flashkda/modal_experiments.py --mode official --heads 96
```

`--mode` 还支持 `profile`、`precision`、`long_precision`、`challenge`、`challenge_profile`、`tile`、`numeric`。NCU report 可用 CPU-only 模式导出：`--mode export --report <baseline.ncu-rep 路径>`。H64 附加形状使用 `--mode official --heads 64`；长序列精度使用 `--mode long_precision --heads 4`。

## 2. 官方 benchmark 复现与语义修正

原始结果：`results/c1-official-h96-20260913T020630921340Z/official-benchmark.json`。严格使用官方 30 warmup、200 iterations × 5 repeats，H=96、D=128，总 T=8192。单位 ms。

| 序列 | FlashKDA BF16 state | no state | FP32 state 接口 | FLA chunk KDA | FLA GDN |
|---|---:|---:|---:|---:|---:|
| 8192 | 1.0785 | 1.0303 | 1.0440 | 2.3699 | 1.2901 |
| 1300/547/2048/963/271/3063 | 0.8888 | 0.8581 | 0.8982 | 2.3809 | 1.3346 |
| 8×1024 | 0.7109 | 0.6952 | 0.7280 | 2.3374 | 1.2841 |

原脚本可运行，但其 FLA 调用早于 `safe_gate` 显式参数。FLA a3edffc 在 `safe_gate=True` 时才选择 `lower_bound * sigmoid(exp(A_log)*(g+bias))` 的有界门。直接照抄原 benchmark 得到的速度对比并非完全相同门函数。因此另外保存一个派生脚本，仅在 KDA 调用中加 `safe_gate=True`，并使用 `state_v_first=True` 的现行名称；原脚本不改。参见 `semantic-matched-benchmark.json` 和 `bench_fwd_safe_gate.py`。

| 序列 | FlashKDA BF16 state | FLA chunk，同门函数 | FLA / FlashKDA |
|---|---:|---:|---:|
| 8192 | 1.0789 | 2.0788 | 1.93× |
| 不等长 6 序列 | 0.8892 | 2.1029 | 2.36× |
| 8×1024 | 0.7110 | 2.0575 | 2.89× |

TASK 引用的 GB200 benchmark 还包括 H64。相同的完整官方程序已在本次 B300 上追加复现，结果见 `results/c1-official-h64-20260913T022812279519Z/`；仍为 D128、总 T8192、30 warmup、200 iterations × 5 repeats，单位 ms。

| H64 序列 | FlashKDA BF16 state | no state | FP32 state 接口 | 原脚本 FLA chunk KDA | FLA GDN |
|---|---:|---:|---:|---:|---:|
| 8192 | 0.9882 | 0.9409 | 0.9541 | 1.6459 | 0.8812 |
| 不等长 6 序列 | 0.6711 | 0.6510 | 0.6740 | 1.6795 | 0.9565 |
| 8×1024 | 0.4827 | 0.4705 | 0.4950 | 1.5767 | 0.8683 |

| H64 序列，同门函数 | FlashKDA BF16 state | FLA chunk | FLA / FlashKDA |
|---|---:|---:|---:|
| 8192 | 0.9882 | 1.4672 | 1.48× |
| 不等长 6 序列 | 0.6704 | 1.4839 | 2.21× |
| 8×1024 | 0.4826 | 1.3869 | 2.87× |

这些是单机本次测量；不能与 GB200 原表作跨硬件因果对比。GDN 是标量 gate 的另一算子，只保留官方表列，不能用作同语义 KDA 正确性或加速比。FP32 state 列只是输入/输出接口精度；内部递推仍为 BF16。官方 benchmark 的初始 state 是很大的 `arange` BF16 值，适合复现原脚本，不代表真实模型状态分布；独立精度实验另用随机状态和门值扫描。

## 3. SASS 与 NCU：不能将整个算子称为一种瓶颈

证据目录：`results/c1-profile-h96-20260913T020442527901Z/`。

- `sass.json` 是实际安装扩展 `cuobjdump --dump-sass` 全文；编译目标 `sm_103a`。
- 存在 `HMMA.16816.F32.BF16` 和 `HMMA.16816.F16`；分别对应 BF16 乘法/FP32 累加与 FP16 Neumann。没有 `HGMMA`、`UTCHMMA`。全二进制静态计数分别 1520 与 24，包含多个模板实例，不能当作一次 kernel 的动态执行次数。
- `baseline.ncu-rep`、`baseline.csv` 为 NCU `--set detailed --clock-control none --cache-control all --launch-count 2`，K1/K2 各 22 passes，退出码 0。旧环境 probe 的 return 9 不能解释为此平台永久不支持 NCU。
- `torch-profile.json` 为独立 5 次调用的 Chrome trace，含 beta transpose 与两个 kernel。

| H96，fixed T8192 | K1 prepare | K2 recurrence |
|---|---:|---:|
| NCU duration | 272.544 μs | 793.440 μs |
| grid / threads | 49,152 CTA / 256 | 96 CTA / 192 |
| 动态 shared memory | 21,248 B | 98,432 B |
| 寄存器 / allocated | 32 / 32 | 65 / 72 |
| shared-memory occupancy limit | 9 CTA/SM | 2 CTA/SM |
| waves/SM | 41.51 | 0.32 |
| DRAM 总吞吐 | 4.529 TB/s | 1.353 TB/s |
| L2 throughput（elapsed peak 比例） | 72.32% | 25.55% |
| tensor active cycles（elapsed） | 5.64% | 18.27% |
| tensor active cycles（active） | 5.73% | 28.40% |
| active warps（active peak 比例） | 96.24% | 9.37% |

K1 显示高并发、高缓存/内存流量、较低 tensor 比例；它还有门函数、归一化、三角矩阵和同步工作，不能只凭总 FLOPs 归为 tensor compute-bound。K2 的 96 CTA 小于 148 SM，长序列内每个 CTA 持续执行 chunk 递推，平均 tensor 与 HBM 都远未全卡饱和；其 shared-memory 配额最多 2 CTA/SM，但实际 grid 不足一个全卡 wave。因而“并行度和依赖链受限”比“全卡 Tensor Core 峰值受限”更符合现有证据。

PC sampling 中 K2 较多的状态为 wait、short_scoreboard、sleeping。它们提示固定延迟依赖、共享内存依赖和等待路径，不能把采样数量直接加成端到端时间，也不能未经逐指令映射把 sleeping 全部归因于某一种 barrier。NCU 多次 replay、cache flush、未锁频与普通 timing 条件不同；其 duration 用于配合 counter 分析，正式速度比使用 benchmark 样本。

TP8 的 H12 对照见 `results/c1-profile-h12-20260913T020932999579Z/ncu-metrics.json`。K1 降至 41.728 μs，K2 却仍为 786.528 μs，接近 H96 的 793.440 μs。K2 的 DRAM 吞吐降至 0.156 TB/s、tensor elapsed 比例降至 2.24%，而 active 比例仍为 28.60%。头数减少 8 倍、K2 时间几乎不变，是每 head 递推链长度与并行覆盖限制的重要交叉证据；这不等于所有模型分布和序列组合都有同样瓶颈。

## 4. 并行度挑战：value-column split

实现源：`challenge_build.py`。在远端 `/opt/FlashKDA` 的副本生成 `flash_kda_split2`，K1 算法与所有算术指令不改。K2 grid 从 `(N,H)` 变为 `(N,H,2)`，每个 CTA 的 compute warp 从 4 减为 2，各 warp 仍负责相同 32 个 value columns。初始状态、workspace 和输入 TMA 复制读取；shared-memory 分配仍为全状态。输出和最终 state 用 store warp 协作写自己负责的 value 范围，包含 varlen tail 和 FP32 state 接口。

正确性依据是状态列之间没有规约：每个 value column 的递推只依赖该列之前的状态、共享 q/k/g/beta，以及对应 v。增加 split 不改变每列的乘加顺序和 BF16 舍入点。该版本刻意保留输入复制与全 shared allocation，衡量增加 grid 覆盖是否足以覆盖复制/写回成本；即使出现负收益，也只否定这个具体实现，不能否定所有 value split 或所有 SM100 专版。

正确性与性能结果见 `results/c1-challenge-h96-20260913T020956599379Z/`。30 组独立精度用例的 split2 output/state 与 baseline 全部逐位相等。另有 8 组 state_in/state_out 存在性 × BF16/FP32 接口组合通过。`results/c1-challenge_profile-h96-20260913T021340887396Z/large-shape-correctness.json` 进一步验证 H96 的三个完整 T8192 benchmark 形状，output/state 全部逐位相等。

性能使用每例 20 warmup，100 iterations × 5 repeats，完整 500 个 CUDA event 样本保存在 `challenge-timings.json`。单位 ms；加速比定义为 baseline/split2。

| H | 序列 | baseline | split2 | 加速比 |
|---:|---|---:|---:|---:|
| 12 | 8192 | 0.8370 | 1.4058 | 0.595× |
| 12 | 不等长 6 序列 | 0.3568 | 0.4751 | 0.751× |
| 12 | 8×1024 | 0.1604 | 0.2293 | 0.699× |
| 96 | 8192 | 1.0761 | 1.8648 | 0.577× |
| 96 | 不等长 6 序列 | 0.8861 | 1.4493 | 0.611× |
| 96 | 8×1024 | 0.7092 | 1.2423 | 0.571× |

这是明确的负结果。split2 的 NCU 报告显示 H96 K2 从 793.440 μs 增至 1,583.040 μs，而 K1 仍约 273 μs。K2 grid 由 96 增至 192，waves/SM 由 0.32 增至 0.65，但动态 shared memory 仍为 98,432 B/CTA；寄存器从 65 增至 80，新增 local-memory load/store 请求分别 1,572,864 / 1,180,416（baseline 为 0）。K2 DRAM 读从 0.886 GB 增至 1.106 GB，重复读取未必全落 HBM，因为缓存能吸收一部分；tensor elapsed 比例反而降至 9.16%。

因此这个实验同时证实两点：列切分在数学和实际精度上可行；仅增加 CTA、保留整块 shared state/输入复制、改写 store warp，会增加访存和寄存器/等待成本，足以吞掉覆盖更多 SM 的收益。现有数据无法将负收益精确分解为每一种代码变化的独立贡献；需要后续消融才可给出该因果分摊。此结果不支持直接发布当前 split2，也不能推导“官方 SM80 是所有 SM100 设计中的全局最优”。

## 5. BF16 状态与独立数值参考

`run_experiments.py::precision` 使用 2 个 seed、H4、长度 16/17/97/1024，以及 varlen 17/33/65；每个长度测试随机 gate、raw gate=-8 且 bias=0 的弱衰减、raw gate=8 且 bias=0 的强衰减，总计 30 组。q/k 为随机归一化 BF16，v/beta 随机，state 为随机 BF16。保存每例 output 和最终 state 的最大绝对误差、平均绝对误差、RMSE、相对 RMSE 与 finiteness。

第一层 oracle 是官方 `tests/torch_ref.py`，复现相同舍入及近似指令口径：30 组 baseline output/state 全部逐位相等。第二层是本地固定版本 `fla_kda_ref/naive.py::naive_recurrent_kda`，每序列分别执行，state 从 value-first 转置为 key-first，再将输出 state 转置回来。输入 q/k 用 FP32 重新归一化，gate 用自然 log 的 bounded sigmoid、beta 用 FP32 sigmoid；该 naive 源码内部显式转 FP32，**不是 FP64 gold**。第三层是禁用 FlashKDA backend 的 FLA Triton chunk，同样启用 safe_gate。

下表给出同 gate 类别、所有长度和 seed 中相对 RMSE 的最大值，单位百分比；FLA chunk 列也相对同一 naive 参考。

| gate 类别 | FlashKDA output | FlashKDA state | FLA chunk output | FLA chunk state |
|---|---:|---:|---:|---:|
| 随机 | 0.5162% | 0.4877% | 0.3385% | 0.2291% |
| 弱衰减 | 0.7213% | 0.7467% | 0.4450% | 0.4190% |
| 强衰减 | 0.5006% | 0.4896% | 0.3099% | 0.1819% |

所有实际结果有限，split2 与 baseline 完全相同，说明挑战没有额外引入数值误差。弱衰减更容易保留早期误差，与理论一致；但这个表测量的是整个实现与独立递推的差异，包含 q/k 舍入、gate 近似、三角求逆、BF16 中间量和 state 舍入。它不能独立证明“全部误差来自 BF16 state”。

进一步对同一弱衰减输入族运行 H4、T8192/32768、seeds 0/1，原始数据见 `results/c1-long_precision-h4-20260913T022617825299Z/precision.json`。4 例 baseline 的 output/state 均与官方 oracle 逐位一致，与独立 naive、FLA chunk 比较的所有结果均有限。下表仍为相对同一 FP32 naive 的 RMSE，单位百分比。

| seed | T | FlashKDA output | FlashKDA state | FLA chunk output | FLA chunk state | FlashKDA output 的最差 1024-token 窗口 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 8192 | 0.745862% | 0.702648% | 0.477221% | 0.407610% | 0.756802% |
| 0 | 32768 | 0.816647% | 0.750500% | 0.483760% | 0.407871% | 0.823819% |
| 1 | 8192 | 0.747068% | 0.702899% | 0.487280% | 0.417300% | 0.758635% |
| 1 | 32768 | 0.752508% | 0.705717% | 0.463419% | 0.392382% | 0.758815% |

这组测试在 32768-token 弱衰减序列中没有出现非有限值或输出窗口误差持续失控的现象；它提供的是明确随机输入族的经验边界，不能保证任意长度、真实 checkpoint 或全部对齐 keys 压力分布的稳定性。不同 T 的随机张量并不共享完整输入前缀，因此不将跨 T 的整体误差差异解读为严格的时间漂移曲线；窗口明细保存在 JSON 中，可核对每条实际序列的局部误差。状态精度与整个算法误差仍应分开讨论。

## 6. 实际 SM80 / tcgen05 tile microbenchmark

源码 `tile_microbench.cu`，最终有效数据目录 `results/c1-tile-h96-20260913T022453478641Z/`，包含编译命令、逐组数据、实际源码快照与 hash、完整 SASS、`tile-summary.json`。两侧均为 BF16 输入、FP32 累加，实际 SASS 分别出现 `HMMA.16816` 与 `UTCHMMA`；tcgen05 侧还有 TMEM load 与 commit/barrier 指令。四个 M/N/K 形状 × 1/12/96/148 CTA × 1/64 次内部重复均通过 CPU 参考检查，最大误差为 0；检查覆盖每个 CTA 的每个输出，预填 NaN 哨兵并显式检查有限性，避免 fmax 忽略 NaN。输入是有限的 dyadic 随机模式，使本组 loops 倍参考可精确核对。

两侧先将 A/B 载入 shared memory，在内部循环累加到**同一个** FP32 accumulator，最后写一次输出。SM80 使用 4 warp 的 `m16n8k16` atom；tcgen05 使用 CTA-group 1，A/B 均 shared memory、D 为 TMEM，只有 issuer warp 推进每次完整 GEMM 的 commit/wait 与 phase；所有其他 warp 在循环末尾 CTA 同步后再读取 TMEM，避免慢 warp 错过复用的 barrier phase。M64 accumulator 使用 16dp64b TMEM load；M128 使用 32dp32b。K16 采用 32-byte swizzle，避免错误套用最低 K64 的 128-byte swizzle atom。接口依据 [NVIDIA CUTLASS SM100 教程](https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/blackwell/01_mma_sm100.cu) 与 [TMEM copy 指令封装](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/copy_sm100.hpp)。

计时使用包含 20 次 kernel 的 CUDA graph，每个样本重放 20 次，除以 400 得单 kernel 平均，5 个重复。为了隔离固定的 global→shared、TMEM 分配/释放、输出和启动成本，另外报告 `(time_64−time_1)/63` 的增量估计；这仍含循环、数据依赖、shared/register 访问和同步，并非单条 MMA 延迟。下表是 96 CTA 的结果，增量单位 ns。

| 形状 M×N×K | C1 对应关系 | SM80 增量 | tcgen05 增量 | SM80/tcgen05 |
|---|---|---:|---:|---:|
| 128×16×128 | 转置后的 KdS / QdS | 182.5 | 242.2 | 0.75× |
| 128×16×16 | 转置后的 INV×R / Mqk×U | 23.2 | 83.5 | 0.28× |
| 128×128×16 | state update | 193.6 | 111.8 | 1.73× |
| 64×16×16 | 小矩阵的 M64 数据通路模型 | 11.4 | 73.5 | 0.16× |

最后一行两侧都真的计算完整 64×16，而实际 C1 inverse 只有 16×16；它验证 M64 数据通路与同步成本，不能声称已与四分之一算术量的原始 inverse 做公平端到端比较。第一、二行证明转置后的 skinny tile 在合法性上可直接执行；第四行则提醒“合法的最小 M64”仍然需要具体 TMEM 数据通路布局。

该微基准对“仅替换指令一定更快”给出反例，同时在 state update 的增量成本上发现正信号。不能从这张表推断 tcgen05 的硬件峰值不如 SM80：本实验每次 GEMM 都 commit/wait，缺少更深流水与跨多个独立 accumulator 的重叠；SM80 的 SASS 显示编译器可以保留不变操作数在寄存器中，因此不声称 SM80 每次循环都重新读取 shared memory。固定 prologue 也没有做性能优化，例如 96 CTA 的 128×128×16 单次完整 kernel 是 SM80 8.632 μs、tcgen05 13.620 μs；64 次重复时分别 20.828/20.665 μs。即使累积 64 次，其完整 kernel 时间仍非常接近；增量优势不能直接当作一次完整 FlashKDA 调用的收益。

早期 `c1-tile-*` 目录保留了宏变量冲突、swizzle/TMEM layout 编译失败，以及独立重复/普通 launch 的方法探索。重复次数不影响真实数据依赖的版本存在编译提升风险，普通 launch 也可能被主机发射间隙主导；独立审查还补齐所有 CTA 有限性检查并修复 barrier phase 复用协议。本节只引用最终全部修正后的依赖链与 CUDA graph 版本，不用早期数据作性能结论。

## 7. C16 / C32 / C64 数值机制

完整独立分析见 `C1_NUMERIC_EXPERIMENTS.md`，实测目录 `results/c1-numeric-h96-20260913T021503565747Z/`。自定义显式 FP16 MMA 的 C16/C32/C64 左右单位阵与随机乘法检查通过，SASS 包含 `HMMA.16816.F16`；共 6 个范围案例、45 个 Neumann 案例。

- 原始 g=-5 指数分解在第 18 个 token（G=-90）出现 FTZ 归零和逆因子上溢，直接扩大 C16 到 C32/64 会遇到 `0×Inf`。直接差分计算可以避免该 NaN，却仍可能把远距离项下溢为 0。
- C16 没有指数范围溢出也不等于最坏数值稳定。对齐 keys 的 β=0.990234375 例，显式 FP16 Neumann 最大绝对逆误差为 1，尽管随机归一化 keys 的例子误差小得多。
- C32/C64 的 Neumann 中间量超出 FP16，且全 FP32 doubling 也可出现严重抵消误差。对 β=1 的 C64 例，矩阵条件数仅 128，FP32 逆误差却约 2.15×10⁹，说明算法中间增长不能用“输入系统必然病态”解释。

这些是独立机制测试，不是 C32/C64 完整 KDA 实现。numeric 报告清楚区分原运行和后续 causal-pair 诊断中额外一次 BF16 乘积舍入的口径修正；原始 JSON/SASS 和可匹配 hash 的实测源码均保留。

## 8. 对 SM100 v2 的实验结论

本次复现支持保留当前 C16/SM80 路径作为可靠默认：H64/H96 六种端到端形状相对相同 bounded gate 的 FLA 有 1.48–2.89× 收益，官方舍入 oracle 与列切分候选都已做实际正确性核对。当前 split2 在全部已测形状负收益，不能直接作为优化提交。

SM100 专版仍有值得研究的切面：state update 的大 M/N tile 在依赖链微基准里显示增量优势；TP8 的 K2 时间对头数不敏感，说明减少每 head 递推临界路径或改进状态列并行值得继续研究。但发布前必须把 TMEM 交互、标量阶段、布局、加载与 store 的代价合并到完整 K2，并以同语义精度和端到端计时证明收益。已有负结果只排除本次具体 split2 和未经流水优化的简单替换，不能为所有可能的 SM100 专版作不可能性证明。

## 9. 对 TASK 六个讨论点的证据索引

| 讨论点 | 本次可支持的结论 | 主要证据 |
|---|---|---|
| 1：为什么 C16 | 原始指数分解 C32/64 越界；Neumann 中间增长与抵消是独立限制；C16 仍有对齐 keys 压力例 | `C1_NUMERIC_EXPERIMENTS.md`、`c1-numeric-*/chunk-numeric.json`、理论报告 §§3–5 |
| 2：只换 tcgen05 | 转置让三个 K2 skinny products 成为合法 M128/N16；本 toy 的 state update 增量有正信号，小 N/同步成本明显 | 本文 §6、最终 `c1-tile-h96-20260913T022453478641Z/` |
| 3：递推之外的并行 | 多 head/CTA、persistent、2-CTA 各有覆盖/同步/冗余反例；本次 value split 实证算术独立可行但简单实现负收益 | 理论报告并行候选章节、本文 §4、`challenge_build.py`、`c1-challenge-*` |
| 4：compute 还是 memory | 必须分别分析 K1/K2；K2 H96→H12 时间近恒定、全卡 tensor/HBM未饱和，支持递推临界路径/覆盖限制 | 本文 §3，H96/H12 `.ncu-rep`、CSV、SASS和Chrome trace |
| 5：BF16 精度 | 官方舍入 oracle 全匹配；独立 naive/FLA、最长 T32768 弱衰减与压力分布揭示误差，不能将所有误差只归因 state 或外推模型质量 | 本文 §5、`c1-challenge-*/precision.json`、`c1-long_precision-h4-20260913T022617825299Z/precision.json`、numeric压力案例 |
| 6：v2 是否专版 | 保留现行可靠默认；当前 split2 不发布；state update/TMEM 流水值得探索但尚无完整 tcgen05 KDA 加速证据 | 本文 §§4/6/8，端到端与微基准分开引用 |
