# C1 首版负结果：实际计数与下一轮可辨别实验

2026-09-17。本文只分析已保存的首版profile和各二进制SASS，不将静态指令数或采样warp状态直接解释为耗时份额。版本、正确性与完整链延迟见 [DESIGN.md](DESIGN.md) 和根目录不可变run。

## 首版 B300 H96、1×8192 的实际 NCU 结果

来源：[profile.csv](../../nextgen/runs/20260916T162554834833Z-c1-profile/artifacts/profile.csv)。该CSV是宽表，一行一个kernel，数值单位位于独立单位行；不能按普通 metric/value 长表解析。

| 指标 | 首版K2实测 | 能说明什么 |
|---|---:|---|
| NCU duration | 13.645152 ms | 与正式计时不同口径，只用于结合counter定位 |
| grid / threads | 96 CTA / 128线程 | 每条状态链一个CTA |
| registers / allocated | 174 / 176 | 不存在本次候选寄存器spill |
| dynamic shared / driver | 163,968 B / 1,024 B | 每CTA合计164,992 B，限制驻留 |
| active warps / active SM | 约4 / 6.249% | 局部等待缺乏可切换的warp |
| issue active，elapsed | 6.580% | 指令发射利用率很低 |
| DRAM总吞吐 | 0.078865 TB/s | 不能解释为全卡HBM带宽饱和 |
| DRAM读取 | 887.216896 MB | 与请求数不是同一计量单位 |
| scalar global load requests | 14,008,320 | 需要结合访问合并与依赖分析 |
| `sm__pipe_tc_cycles_active`，elapsed / active | 0.705723% / 1.114728% | 保留NCU原始管线命名，不将其直接换算为tcgen05峰值利用率 |
| `sm__pipe_tensor_cycles_active`，elapsed / active | 0.265428% / 0.419259% | 不是tcgen05独立计数，不能单凭此项判断tcgen05利用率 |
| shared实际 / 理想wavefronts | 408,847,296 / 232,686,528 | 有176,160,768额外wavefronts，约75.7%额外工作 |
| TMEM load执行数 | 37,748,736 | 恰为192×4warp×96CTA×512chunk |

`derived__memory_l1_conflicts_shared_nway=1567` 是派生/聚合值，没有逐条访问上下文，不能报告成“1567-way bank conflict”。计数单位、动态次数、per-SM值和全卡值不可混用。表内两条管线指标分别来自 `sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_elapsed/active` 与 `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed/active`，必须保留其区别；尤其不能把较低的 `pipe_tensor` 数值称为tcgen05专属利用率。涉及tcgen05的结论需联合实际SASS、专用指令/管线计数和完整链测量。

PC采样595,247条，2个采样pass，dropped bytes与buffer overflow均为0。其主要状态分布：

| warp状态 | 样本数 | 占采样比例 |
|---|---:|---:|
| long scoreboard | 203,622 | 34.21% |
| short scoreboard | 132,283 | 22.22% |
| wait | 124,142 | 20.86% |
| selected | 62,041 | 10.42% |
| no instructions | 30,314 | 5.09% |
| MIO throttle | 17,227 | 2.89% |
| barrier | 13,036 | 2.19% |
| branch resolving | 11,794 | 1.98% |

这些是采样warp的状态，不是wall-time分解。long scoreboard支持检查global load依赖；short scoreboard与wait提示片上结果消费/依赖；要定位到具体TMEM、shared或同步指令仍需要PC与SASS对应，不能把三者相加后宣布某段代码占总时间77%。目前较稳妥的解释是：少量驻留warp下，大量指令、内存请求和跨阶段依赖暴露在关键路径上，整卡算力及DRAM带宽没有饱和。

## 已完成的布局变体没有解决主要问题

首版先将TMEM结果写到row-major FP32 scratch，再由另一种线程分配读取。静态SASS发现第一/第二/第三阶段分别8/4/36条 `STS.128`。按128B子事务推导，N32写回约8-way，N16/N144约4-way冲突；这是对具体地址模式的推导，实际NCU确认总体存在大量额外shared工作，但尚未把176,160,768全部分配到这些PC。

`direct-rowmajor` 删除FP32 scratch、直接在寄存器执行数值epilogue，保留原BF16 backing；`direct` 进一步改成value维连续的BF16 backing。两者都通过独立38例：[rowmajor验收](../../nextgen/runs/20260916T162424899590Z-c1-verify/artifacts/verify-direct-rowmajor.json)、[direct验收](../../nextgen/runs/20260916T162424917086Z-c1-verify/artifacts/verify-direct.json)。六形状几何平均baseline/candidate速度比分别仅约0.085133×和0.078748×，见 [rowmajor计时](../../nextgen/runs/20260916T162640146933Z-c1-bench/artifacts/bench-direct-rowmajor.json)、[direct计时](../../nextgen/runs/20260916T162640192436Z-c1-bench/artifacts/bench-direct.json)，首版为0.071752×。它们只是相对于首版略快，依然远慢于原版；不能写成算子优化已经成功。

两个v2都使用239个寄存器、0spill。SASS的静态计数如下（包含整个K2函数中的分支及冷路径）：

| 静态站点 | 首版fused | direct-rowmajor | direct |
|---|---:|---:|---:|
| LDTM | 192 | 192 | 192 |
| WARPSYNC | 214 | 214 | 214 |
| CALL | 213 | 213 | 213 |
| STS | 127 | 87 | 227 |
| LDS | 290 | 273 | 385 |

direct使按value维访问更连续，却可能损失每线程连续元素的向量化；较多STS/LDS为这个假设提供线索，不能直接当作运行时间归因。更醒目的共同问题是192条逐列TMEM读取及其控制开销完全未变。

## 三项严格分开的后续消融

1. `direct → direct-release`：同一个 `k2_tcgen05_direct.cuh`，只增加 `-DNDEBUG`。当前SASS中每个TMEM copy都有warp/DP匹配检查、条件分支与冷路径assert调用。实验先确认这些代码是否消失，再比较完整forward及K2诊断；不能预先假定全部CALL在运行时执行。`-DNDEBUG`作用于整个CUDA翻译单元，也可能改变K1检查，所以解释K2机制时使用K2-only和按kernel拆开的profile。
2. `direct-release → direct-wide`：同样启用 `-DNDEBUG`，只把TMEM复制原语从 `SM100_TMEM_LOAD_32dp32b1x` 换成 `SM100_TMEM_LOAD_32dp32b16x`。三个N为16/32/144，都被16整除。identity坐标从同一个新copy的 `partition_D` 推导，不复用旧copy的硬编码lane规则。build必须先在固定CUTLASS源码中核对确切struct、16个DRegisters和对应PTX，并保存片段/hash；未经这个核对不能宣称该固定版本支持新原语。
3. `direct-wide → direct-wide-rowmajor`：使用逐字相同的wide header与相同 `-DNDEBUG`，仅将 `C1_TRANSPOSE_SHARED=1` 改成0；重新比较旧BF16 row-major backing是否在宽读取条件下仍更快。这个变体同样先执行pinned原语审计；未实测前只表示实验入口已就绪，不继承direct-wide或旧direct-rowmajor的正确性/性能结论。

若16x复制保持相同元素量而成功减少静态LDTM站点，预期三阶段分别变为2/1/9组，即12组；实际SASS可能将PTX拆解，最终以真实二进制为准。只有完整38例和实际计时输出通过旧exact门槛才能获得正式性能资格。`run.py`资格复用同时核对source、launcher、variant、arch与完整nvcc flags，不能把debug版本PASS自动赋给release。

此时尚未实现persistent状态的原生UMMA布局、TMA预取或线程角色流水化。如果release与wide仍不足，下一步应先看新profile，判断是否还值得消除每chunk从普通BF16 shared到UMMA swizzle的额外复制。不能跳过实测，直接把更多机制叠加到同一个候选后再猜哪个有效。
