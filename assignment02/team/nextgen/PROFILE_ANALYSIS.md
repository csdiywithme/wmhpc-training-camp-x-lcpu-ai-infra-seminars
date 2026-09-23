# 数据流瓶颈：静态推导、NCU 证据与 v2 检验

记录日期：2026-09-17。下述数字来自冻结 run；后续结果以对应 run 的源快照和原始记录为准。本审查没有运行 GPU，也没有修改 kernel 或验收门槛。

## C1 v1：问题首先出在结果搬运与消费

[首版源码][c1-source]使用三个 MMA：`(M,N,K)=(128,32,128),(128,16,16),(128,144,16)`。TMEM 结果先写 row-major FP32 `s.acc`，再由另一种线程分工读取、转换为 BF16。最大 scratch 为 `128*144*4=73,728 B`；三个结果每 chunk 共 `128*(32+16+144)*4=98,304 B`，即约 **192 KiB shared 写入加读取**，不包括其他操作数和状态流量。

源码计算的 dynamic shared 为 **163,968 B/CTA**。实际 [resource dump][c1-resource]还记录 static shared 1,024 B、174 registers/thread、STACK/LOCAL 均为 0；不能把首版慢归因于已经发生的寄存器 spill。

[实际 SASS][c1-sass]的 candidate kernel 包含 192 条静态 `LDTM`、48 条 `STS.128`，以及真正的 `UTCHMMA`。这些是静态指令数量，包含编译器生成的不同控制路径，不是运行时发射次数。结果确实经过硬件 tensor core；问题不是新指令被标量乘加模拟。

TMEM load 分区让连续 lane 持有连续 row。固定结果列 `c` 时，FP32 scratch 地址为 `4*(row*N+c)`；按 32 个四字节 bank 模型，bank 为 `(row*N+c)%32`。官方依据是 [CUDA 13.1 shared memory access patterns][cuda-shared]。

| 结果 N | 相邻 row 字节跨度 | SASS 地址证据 | 对 STS.128 的静态预测 |
|---:|---:|---|---|
| 32 | 128 | PC `9560` 乘 `0x80`，`95b0` 存 `[R3+0x16000]` | 每 128B 子事务按 8 lanes、每 lane 4 floats，预计 8-way |
| 16 | 64 | PC `ccd0` 左移 6，`cce0` 存 scratch | 同模型预计 4-way |
| 144 | 576 | PC `1b1a0` 乘 `0x90`，`1b1e0` 左移 2，`1b200` 存 scratch | 同模型预计 4-way；该阶段有 36 条 STS.128 |

这里的 4/8-way 是按指令分解及地址分布推导的假设。尚未用逐 PC NCU 数据确认，不能当成该 kernel 的已测平均冲突倍数，更不能直接乘成端到端减速比。

V 读取、output 写入及 `kr` 转置读取还有另一种问题。源码循环 `i=tid+128*j`、`token=i%16`、`value=i/16`，因此 warp 内 `token=lane%16`、`value=8*j+2*warp+floor(lane/16)`。V 地址为 `base+2*(token*H*128+value)`：32 lanes 访问 16 个 token 行，每行两个相邻 BF16，共 64B useful，却触及 16 个 32B sectors；完全连续且对齐只需 2 个 sectors。`kr` 的 token stride 为 256B，也呈同样分散访问。

这意味着**单条 warp 请求**的 sector 利用率约 12.5%，不是已经测得 8 倍 DRAM 浪费。跨循环和跨 warp 的 cache 复用可能重复使用已取回的数据；编译器实际访存宽度也需结合 SASS 确认。[CUDA 官方 coalescing 说明][cuda-coalescing]同时解释了 32B 事务和 cache 复用的影响。

另一项独立 codegen 观察：SASS PC `67a0–67f0` 有 warp/DP 检查、`LDTM` 和条件分支，错误分支会调用 assert。它与 [CuTe TMEM copy 的调试断言][cute-tmem]匹配；v1 编译参数未显式指定 `-DNDEBUG`。尚未获得该项单独消融，不能宣称它解释了主要时延；链接为官方主线，实际编译版本仍以 run manifest 的 pinned commit 为准。

## C1：随后取得的真实 NCU 与 v2 负结果

[C1 profile CSV][c1-profile]对应 B300、v1 fused、grid `(1,96,1)`、128 threads/CTA；GPU 有 148 SM。CSV 第二行是 units，`gpu__time_duration.sum` 的 **13.645152 单位是 ms**，不是 µs。

| C1 v1 K2 指标 | NCU 值 | 可支持的解释 |
|---|---:|---|
| `launch__shared_mem_per_block` | 164.992 Kbyte | 163,968 dynamic + 1,024 static bytes |
| `launch__occupancy_limit_shared_mem` | 1 block | shared 资源上限；不是所有场景的实际 occupancy |
| `launch__registers_per_thread` | 174 | 高寄存器占用；resource dump 未见 local spill |
| `derived__memory_l1_wavefronts_shared_excessive` | 176,160,768 | shared 搬运确有大量额外 wavefront，尚未逐 PC 归因 |
| `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum` | 14,008,320 | global load 请求数量，不等于 DRAM transactions |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | 6.249320% | 活跃 warp 很少；96 CTA 的规模及单 CTA 资源均需考虑 |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | 6.580377% | 不能仅凭换用 tensor core 宣称接近峰值 |

同次 source counters 的 shared 实际值为 **408,847,296**、ideal 为 **232,686,528**，差值正好为 176,160,768；原 CSV 将两者单位标成 `sectors`，这里保留该口径，不把差值解释为 DRAM 字节。动态 `smsp__sass_inst_executed_op_tmem_ldt.sum=37,748,736`，恰为 `192 loads/warp/chunk * 4 warps * 96 CTA * 512 chunks`，与首版反复逐列读取 TMEM 的数据流吻合。

PC sampling 共 595,247 samples、2 passes，dropped bytes 和 buffer overflow 都是 0。`long_scoreboard=203,622`（34.21%）、`short_scoreboard=132,283`（22.22%）、`wait=124,142`（20.86%）是**采样占比**，不是可相加的精确耗时分解或可兑现加速比例。它们支持继续定位内存依赖与指令等待，不能独自指定哪一条源码必然占了同样比例时延。

平均 active warps 为 3.999565，issue active 为 6.580377%；DRAM 吞吐仅 0.078865 Tbyte/s。`pipe_tensor` elapsed/active 分别 0.265428%/0.419259%，而 Blackwell UTC*MMA 对应的独立 `pipe_tc` 为 0.705723%/1.114728%。[官方管线分类][ncu-pipelines]区分这两类，不能用旧 tensor 指标单独估计 tcgen05 的 FLOP 利用率。这里更像低并行、等待和片上结果搬运的问题，证据不支持“全卡算力或 DRAM 带宽已饱和”。

该 CSV 的 `derived__memory_l1_conflicts_shared_nway=1567` 也不是“1567-way bank conflict”；这是导出的汇总值，必须回到 source/PC 明细及正确的计数分母解释。

[v1 bench][c1-bench]是完整 forward 的 eager CUDA-event 计时，包含 beta transpose，排除 workspace 分配；6 形状对旧版的 speedup 为 0.063–0.086×，几何均值 **0.071752×**。K2 单独测量约占完整候选时延的 97–99%，因此优先优化 K2 有依据，但两种测量不可直接当作可相减的精确分解。

v2 `direct-rowmajor` 去掉 FP32 scratch；`direct` 还把 BF16 state/U/base_out 改为 value 连续布局。两者既有 38 项 same-rounding suite 均 PASS（JSON 标识为 `suite=quick`），不扩称未测试域已经覆盖。新源码见 [direct epilogue][c1-direct-source]。

| 完整 forward 候选 | 6 形状 speedup 几何均值 | 相对 v1 的归一化比值 |
|---|---:|---:|
| v1 fused | 0.071752× | 1.000× |
| [direct-rowmajor][c1-rowmajor-bench] | 0.085133× | 1.186× |
| [direct][c1-direct-bench] | 0.078748× | 1.098× |

每次 run 内都与旧版配对；上表跨 run 比值是描述性归一化比较，不是三个候选在同一次会话中随机交错的严格因果估计。`direct` 在全部 6 个 case 都慢于 `direct-rowmajor`。**去 scratch 有小幅收益；“仅靠这些改动就恢复旧版竞争力”被当前完整 forward 结果否定。** value 连续的局部地址优势不足以预测整体更快，重新打包 A 的成本、寄存器代码生成与状态串行链仍待定位。

## C2：真实 profile 160153199885Z

[profile.json][c2-profile-json]为 B300、TP1/B1/BF16/seed101、16 splits、feature merge、PDL off、无 fallback；NVTX 内只有一次完整 `partial+merge`，5 次 warmup 和独立 gold 校验在外。前后输出均满足既有 frozen family bounds。原始 [NCU report][c2-report]、[CSV][c2-csv]及[提取值][c2-selected]均保留。

| NCU 指标 | tcgen05 partial | feature merge |
|---|---:|---:|
| `gpu__time_duration.sum` | 81.728 µs | 4.960 µs |
| registers/thread | 70 | 39 |
| shared/block（原始单位 Kbyte） | 58.624 | 1.152 |
| shared occupancy limit（block） | 3 | 56 |
| active warps / peak sustained active | 6.268030% | 1.708224% |
| SM throughput / peak sustained elapsed | 2.146915% | 0.558891% |
| shared excessive wavefronts | 413,696 | 0 |
| `derived__memory_l1_conflicts_shared_nway` | 353 | 5 |
| theoretical global excessive sectors（导出单位 Mbyte） | 1.040384 | 0.001536 |
| global load requests | 70,400 | 1,152 |
| DRAM bytes read（Mbyte） | 4.255488 | 0.272896 |

**353 不能读成一次访问的 353-way 冲突，也不能称为 warp 平均冲突倍数。** 当前 CSV 只有 kernel 汇总，缺少对应逐 PC 分布与聚合分母；merge 的该值为 5 而 excessive wavefronts 为 0，也说明不能用这个单值取代完整访问分析。单位为 Mbyte 的 excessive sectors 字段是导出的理论量，不是实测 DRAM 多传了相同字节。

partial grid 只有 `(16,4,1)`，共 64 CTA，设备有 148 SM；shared limit=3 不能证明实际每 SM 驻留 3 CTA。低活跃率同时受任务规模、依赖、同步和资源限制影响。该证据支持先检查 partial 数据流，却不能独自证明 shared 是唯一瓶颈。[官方 NCU profiling guide][ncu-guide]区分 request、sector、wavefront 和 occupancy 的含义。

本次命令设置 `--clock-control none`，保留 NCU 默认 cache/replay 行为；仪器时延 **81.728+4.960 µs 不是 hot-Graph benchmark 的链路时延**。同形状 seed 的[独立 paired bench][c2-bench]测得 baseline 5.681548 µs、merge-only 5.057032 µs、tcgen05 完整链 43.828014 µs。NCU replay、cache 控制、kernel 序列化和计时边界会改变观测条件；官方对此有[明确说明][ncu-duration]。

## 两条可以被后续实测推翻的 v2 预测

1. **C1：去 scratch 降低 K2 成本，但 value 连续布局未必继续降低完整时延。** 已执行的区分实验是 v1→direct-rowmajor→direct；源级预期减少约 192 KiB/chunk 的 scratch 往返，dynamic shared 降到约 90,240B，不能推出 occupancy 或速度翻倍。现有结果支持第一步的有限收益，否定第二步在这 6 形状上的更快预测。若后续同会话配对不能复现第一步的 K2/完整 forward 降幅，应撤回“scratch 是有效优化点”的性能归因；必须同时检查实际寄存器和逐 PC shared 指标，不能只看源码少了 buffer。
2. **C2：coalesced feed 降低 global sector 请求浪费，pad17 在其基础上降低 softmax 的 shared 额外 wavefront，并应降低完整 partial+merge 中位时延。** [首版 softmax][c2-source]固定 head 时 stride16 的 bank=`(16*token+h)%32` 仅用两个 bank；stride17 则遍历 32 个 bank。用 original→coalesced→coalesced_pad17 逐项对照，保持相同输入、split、merge、precision 和 frozen gate。若 stride17 修复合法性后 shared 指标不降，则局部映射/归因预测失败；若指标降而完整链不快，则“该冲突主导端到端性能”的预测失败。第一版 pad17 已出现 misaligned address，默认 128-bit shared copy 的对齐假设需修正；[失败 run][c2-pad17-failure]必须保留，修复版通过完整正确性前不产生性能结论。

这两条都要求完整 forward/attention chain；单 tile、单页或 profile 中局部时长改善不能替代最终验收。

## C2 coalesced 变慢后：是否只是把 global 浪费换成 shared 冲突？

[coalesced 正式 32-case bench][c2-coalesced-bench]相对旧版的几何均值为 **0.073612×**，低于 original 的 **0.093990×**；TP1/B1/BF16/seed101 完整链从 43.828014 µs 变为 55.305199 µs（分别来自各自 paired run）。这证明当前实现没有带来整体收益，尚不能反推某一种冲突是原因。

按 [pinned SW32 原子][cute-sw32]和 MMA 的 `m` 最快分区推导，BF16 K-major atom 为 8×16；扩展到 M×128 后，未 swizzle 的相对 byte offset 为 `b=32*m+2*(k%16)+32*M*floor(k/16)`，物理地址再作 `b XOR ((b>>3)&16)`。实际基址的 swizzle 相位可能置换 bank 编号，不改变下表的重复度。表中 way 仅按一条标量 BF16 warp store 的**不同 32-bit words/bank**计算，尚未通过 pinned host 地址打印及逐 PC SASS/NCU 验证；编译器向量化会改变事务划分。

| 供数 | original 的 warp 逻辑坐标 | coalesced 的 warp 逻辑坐标 | BF16 global sectors / request（连续 backing） | shared 标量模型 |
|---|---|---|---|---|
| K | 连续 token、固定 d | 固定 token、连续 d | 32 → 2 | 8 banks×4 words → 8 banks×2 words |
| Q | 16 heads 各两个相邻 d | 固定 head、连续 d | 16 → 2 | 8 banks×2 words → 8 banks×2 words |
| V | 固定 token、连续 d | 固定 token、连续 d | 2 → 2 | 两者都是 8 banks×4 words |

尤其 V：original `ca(i)=(i%128,i/128)`，而新代码先令 `token=i/128,d=i%128`，再写 `sa(inverse_a(d*128+token))`；在该分区下逆映射恢复的 ordinal 正好还是 `i`。所以**新旧 V 的逐线程 global→shared 地址相同**是当前静态预测；转置 scatter 的成本一直存在，不能把它称为 coalesced 新引入的恶化。`pad17` 只改 score/P temporary 和合法 TMEM 回写，不改变 K/Q/V 的 operand 布局。

[coalesced resource dump][c2-coalesced-resource]另有真实 codegen 变化：BF16 registers/thread 从 70 增到 **102**，FP8 从 70 增到 **72**，LOCAL/STACK 仍为 0。寄存器增长提示应检查指令安排与活跃值，但 shared 容量未变，不能据此宣称 spill 或驻留数必然下降。

下一轮保持同一输入、split、merge、PDL 与 NCU 配置，比较 original/coalesced/pad17 的 **partial**，优先三项：

1. `derived__memory_l2_theoretical_sectors_global_excessive`：检查 K/Q 请求浪费是否下降；同时保留 raw actual/ideal 和 units，v1 为 1.040384 Mbyte。它不是 DRAM 流量。
2. `derived__memory_l1_wavefronts_shared_excessive`：v1 为 413,696；按 K feed、V feed、softmax、TMEM 回写的 source/PC 分组，区分供数 scatter 与 stride17 的收益，不能只比总和后猜原因。
3. `smsp__inst_executed.sum`：v1 为 2,095,104；若 global 指标改善而动态指令量/时延增大，优先查 inverse 地址计算、循环展开和 copy 指令宽度。三个指标都不能替代完整链的独立 paired benchmark。

## C2 coalesced 的实测证伪（000536967954Z）

已核对新增[带 units 的 NCU 提取记录](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/nextgen/runs/20260917T000536967954Z-c2-profile/artifacts/profile-comparison-selected.json)及原始 CSV。同一 TP1/B1/BF16/seed101 的 partial，global excessive 从 **1.040384 Mbyte 降到 0 Kbyte**，shared excessive 从 **413,696 降到 348,160（−15.84%）**；动态指令从 **2,095,104 增至 2,153,280（+2.78%）**，global LD 请求仍为 **70,400**，TMEM load 仍为 **8,192**。因此确实改善了这两个访问计数，不能用“新 shared 总冲突恶化”解释整体变慢。

与此同时 partial 的 NCU duration 从 **81.728 µs 增至 131.680 µs**，独立完整 bench 也更慢（见上节）。这组结果否定“改善 global 合并和 shared excessive 就足以改善当前完整链性能”的预测；**精确主因仍未确定**，+2.78% 指令量或寄存器增长本身都不足以量化解释延迟增长，仍需指令安排、依赖链及逐 PC 分析。

寄存器 70→102 对应 register occupancy limit 从 7→4 blocks，但 shared limit 两次均为 3，grid 两次均为 64 CTA；实测 active warps/active SM 约 **4.011539→3.998966**。故不能声称“102 registers 导致实际 occupancy 明显下降”。提取文件按列自适应单位：merge 的 **1.536 Kbyte 等于旧值 0.001536 Mbyte**，没有 1000 倍增长；上述 NCU duration 始终只用于机制诊断，不替代 Graph benchmark。

[c1-source]: /Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/nextgen/runs/20260916T155457579434Z-c1-bench/sources/c1/k2_tcgen05.cuh
[c1-resource]: /Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/nextgen/runs/20260916T155457579434Z-c1-bench/artifacts/binary-0-resources.json
[c1-sass]: /Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/nextgen/runs/20260916T155457579434Z-c1-bench/artifacts/binary-0.sass
[c1-profile]: /Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/nextgen/runs/20260916T162554834833Z-c1-profile/artifacts/profile.csv
[c1-bench]: /Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/nextgen/runs/20260916T155457579434Z-c1-bench/artifacts/bench-fused.json
[c1-direct-source]: /Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/nextgen/runs/20260916T162640192436Z-c1-bench/sources/c1/k2_tcgen05_direct.cuh
[c1-rowmajor-bench]: /Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/nextgen/runs/20260916T162640146933Z-c1-bench/artifacts/bench-direct-rowmajor.json
[c1-direct-bench]: /Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/nextgen/runs/20260916T162640192436Z-c1-bench/artifacts/bench-direct.json
[c2-profile-json]: /Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/nextgen/runs/20260916T160153199885Z-c2-profile/artifacts/profile.json
[c2-report]: /Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/nextgen/runs/20260916T160153199885Z-c2-profile/artifacts/profile.ncu-rep
[c2-csv]: /Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/nextgen/runs/20260916T160153199885Z-c2-profile/artifacts/profile.csv
[c2-selected]: /Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/nextgen/runs/20260916T160153199885Z-c2-profile/artifacts/profile-selected.json
[c2-bench]: /Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/nextgen/runs/20260916T155513744579Z-c2-bench/artifacts/paired.json
[c2-source]: /Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/nextgen/runs/20260916T160153199885Z-c2-profile/sources/c2/nextgen/partial.cu
[c2-pad17-failure]: /Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/nextgen/runs/20260916T162424950978Z-c2-smoke/state.json
[c2-coalesced-bench]: /Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/nextgen/runs/20260916T235840124691Z-c2-bench/artifacts/paired.json
[c2-coalesced-resource]: /Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/nextgen/runs/20260916T235840124691Z-c2-bench/artifacts/binary-0-resources.json
[cute-sw32]: https://github.com/NVIDIA/cutlass/blob/5c149f52a436782210263fb2f19b354443a61c6a/include/cute/atom/mma_traits_sm90_gmma.hpp#L75
[cuda-shared]: https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/02-basics/writing-cuda-kernels.html#shared-memory-access-patterns
[cuda-coalescing]: https://docs.nvidia.com/cuda/archive/13.1.0/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory
[cute-tmem]: https://github.com/NVIDIA/cutlass/blob/main/include/cute/atom/copy_traits_sm100.hpp
[ncu-guide]: https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#quantities
[ncu-duration]: https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#workload-durations
[ncu-pipelines]: https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#pipelines
