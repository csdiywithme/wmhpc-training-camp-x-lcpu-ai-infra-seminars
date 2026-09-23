# C1 完整K2负结果：首版、最终候选与原版的实际计数

2026-09-17。本文分析已保存的首版、最终候选、原版profile与二进制SASS，不将静态指令数或采样warp状态直接解释为耗时份额。版本、正确性与完整链延迟见 [DESIGN.md](DESIGN.md) 和根目录不可变run。最终候选保持旧语义，但六形状完整forward仍比原版慢约6.97–9.46倍；本轮停止内核修改，将剩余机制作为下一轮的可辨别假设。

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

## 三项已完成、严格分开的消融

1. `direct → direct-release`：同一个 `k2_tcgen05_direct.cuh`，只增加 `-DNDEBUG`。当前SASS中每个TMEM copy都有warp/DP匹配检查、条件分支与冷路径assert调用。实验先确认这些代码是否消失，再比较完整forward及K2诊断；不能预先假定全部CALL在运行时执行。`-DNDEBUG`作用于整个CUDA翻译单元，也可能改变K1检查，所以解释K2机制时使用K2-only和按kernel拆开的profile。
2. `direct-release → direct-wide`：同样启用 `-DNDEBUG`，只把TMEM复制原语从 `SM100_TMEM_LOAD_32dp32b1x` 换成 `SM100_TMEM_LOAD_32dp32b16x`。三个N为16/32/144，都被16整除。identity坐标从同一个新copy的 `partition_D` 推导，不复用旧copy的硬编码lane规则。build必须先在固定CUTLASS源码中核对确切struct、16个DRegisters和对应PTX，并保存片段/hash；未经这个核对不能宣称该固定版本支持新原语。
3. `direct-wide → direct-wide-rowmajor`：使用逐字相同的wide header与相同 `-DNDEBUG`，仅将 `C1_TRANSPOSE_SHARED=1` 改成0；重新比较旧BF16 row-major backing是否在宽读取条件下仍更快。这个变体同样先执行pinned原语审计，并重新取得38例资格，没有继承direct-wide或旧direct-rowmajor的正确性/性能结论。

实际二进制确认16x复制保持元素量、三阶段LDTM站点分别为2/1/9，共12处；原1x版本为192处。release与wide两版的K2静态WARPSYNC/CALL分别为19/3，较debug版本214/213明显减少；若统计整个binary，WARPSYNC对应为29与224，应保持统计范围一致。这些不是运行时执行次数或冷路径耗时。三变体都通过完整38例和实际计时后的旧exact门槛，六形状几何平均速度比分别0.095781×、0.104891×、0.120046×。`run.py`资格复用同时核对source、launcher、variant、arch与完整nvcc flags，不能把debug版本PASS自动赋给release。

本轮尚未实现persistent状态的原生UMMA布局、TMA预取或线程角色流水化。以下新profile说明release与wide远不足以接近原版，也说明删除一个scratch不等于消除整个递推的shared冲突。

## 最终候选与首版：工作减少了，但冲突比例反而更高

来源：[最终候选profile](../../nextgen/runs/20260917T001710083225Z-c1-profile/artifacts/profile.csv)，同为B300、H96、1×8192。表内只有K2；NCU时间是诊断口径，不替代正式成对计时。

| 指标 | 首版fused | 最终direct-wide-rowmajor |
|---|---:|---:|
| NCU duration | 13.645152 ms | 8.038112 ms |
| registers，0spill | 174 | 168 |
| dynamic shared / 含driver合计 | 163,968 / 164,992 B | 90,240 / 91,264 B |
| grid / block线程 | 96 / 128 | 96 / 128 |
| shared容量上限 | 1 CTA/SM | 2 CTA/SM |
| active warps | 3.999565 | 4.000045 |
| warp instructions | 1,013,836,704 | 564,750,720 |
| TMEM load执行数 | 37,748,736 | 2,359,296 |
| shared理想wavefronts | 232,686,528 | 128,877,504 |
| shared额外wavefronts | 176,160,768 | 254,803,968 |
| shared实际wavefronts | 408,847,296 | 383,681,472 |
| scalar global load指令计数 | 14,008,320 | 41,730,048 |
| DRAM读取 | 887.216896 MB | 885.712384 MB |
| `pipe_tc` elapsed / active | 0.705723% / 1.114728% | 1.243193% / 1.948438% |
| `pipe_tensor` elapsed / active | 0.265428% / 0.419259% | 0.450770% / 0.706485% |

实际TMEM执行数恰好减少16倍，与12×4warp×96CTA×512chunk相符；动态warp指令减少约44.3%。去掉73,728B scratch降低了所需shared容量与理想事务数，但额外wavefronts增加约44.6%，实际shared总工作只降低约6.2%。因此“scratch已去掉，所以shared冲突已解决”与实测矛盾。容量允许2CTA/SM也没有在96CTA的小grid上自动提供更多活跃warp，平均值仍约4。

最终PC采样353,349条，2pass、dropped bytes及overflow均为0。long scoreboard为178,253（50.45%），short scoreboard67,367（19.07%），wait59,519（16.84%），selected34,071（9.64%），barrier7,050（2.00%）。long scoreboard占比上升且global load计数增多，支持继续检查标量加载依赖；其比例不是wall-time占比。DRAM字节数基本不变，吞吐仅0.133694TB/s，不能把较慢归因为HBM总带宽饱和。

## 为什么直接BF16 epilogue仍会冲突

最终候选 [实际SASS](../../nextgen/runs/20260917T001710083225Z-c1-profile/artifacts/binary-0.sass) 与 `k2_tcgen05_direct_wide.cuh` 的 `location<K>` 一致：TMEM consumer的相邻lane对应相邻value行，row-major BF16 state地址是 `2*(value*128+key)`，U/base地址是 `2*(value*16+token)`。宽TMEM load合并了读取列，但没有自动改变BF16 shared的lane访问布局。

可以落到实际指令的静态证据如下；冲突倍率按32个4B bank和128B子事务推导，尚未获得逐PC硬件wavefront归属，不能把总计数全部分摊到这些位置。

| 路径 | 实际指令和地址证据 | 访问推导 |
|---|---|---|
| 状态FMA的旧state读取与新state写回 | PC `a790` 为 `LEA R0,R149,UR15,0x8`，随后 `a7b0 LDS.128 [R0]`、`aa60 STS.128 [R0]`；每行256B | 每8lane的128B子事务落相同4个bank但不同word，约8-way；不能用标量模型称32-way |
| 首投影残差写s.u | PC `7a30` 为行地址乘32B；`8040–8140` 为16条 `STS.U16`，列偏移0至30B | 固定token时32lane只使用4个bank，每bank有8个不同word，约8-way |
| rounded U和base_out写回 | `8750/87e0`、`7bb0/8030` 为 `STS.128`，同32B行距 | 各128B子事务8lane只覆盖16个bank，约2-way；向量化减轻但没有清除冲突 |

state更新的16次 `LDS.128` 和16次 `STS.128` 如按上述模型逐chunk执行，仅这一模式就可能形成大量额外wavefronts；这是用于下一轮挑选实验的访问模型，不是已有的逐PC实测归因。另外仍存在普通BF16 backing→UMMA swizzle的合作复制，以及base_out读取等路径，删除FP32 scratch没有删除它们。

首版与最终候选的shared反例也说明，直接换成简单列优先布局不能作为充分解决方案：已测 `direct` 和 `direct-wide` 都比各自row-major对照慢，布局还会改变每线程向量化、指令数量与合作复制。下一轮应同时设计epilogue映射和MMA操作数的物理布局，再用独立counter与端到端计时判定，不能只根据一张bank映射图宣布更快。

## 同grid原版对照：主要差距在数据流，而非新指令本身的算力

来源：[原版profile](../../nextgen/runs/20260917T001950764975Z-c1-profile/artifacts/profile.csv)。注意此CSV的duration单位为μs，候选CSV为ms；原版K2是794.272μs，不能写成794ms。两者都使用同一H96、1×8192输入形状。

| K2指标 | 原版 | 最终候选 |
|---|---:|---:|
| NCU duration | 794.272 μs | 8,038.112 μs |
| grid | 96 CTA | 96 CTA |
| block线程 / warp | 192 / 6 | 128 / 4 |
| registers / thread | 65 | 168 |
| shared含driver | 99,456 B | 91,264 B |
| shared容量上限 / waves | 2 / 0.32 | 2 / 0.32 |
| active warp / 活跃率 | 5.996445 / 9.369446% | 4.000045 / 6.250070% |
| issue active，elapsed | 20.602675% | 6.225113% |
| warp instructions | 184,307,697 | 564,750,720 |
| shared额外wavefronts | 0 | 254,803,968 |
| scalar global load指令计数 | 0 | 41,730,048 |
| DRAM读取 | 885.587200 MB | 885.712384 MB |
| `pipe_tc`，elapsed | 0% | 1.243193% |
| `pipe_tensor`，elapsed | 18.254387% | 0.450770% |

原版源码 `fwd_kernel2.cuh` 中4个compute warp外另有TMA load/store各1个warp，输入通过pipeline进入已适配MMA的shared布局；输出也走TMA。因此原版scalar global load计数为0不表示不读显存，而是这项指标不计TMA。两版实际DRAM读取量相差约0.014%，候选却通过41.73m次标量load完成搬运/直接epilogue加载，其中gate `p_gt[key]` 在不同value消费者间重复读取。其实际SASS在 `a690` 之后有大量相同gate基址、不同key偏移的 `LDG.E`；缓存可减少外部字节，却不能免费消除指令与结果依赖。

原版与候选同grid、同shared容量驻留上限，候选甚至shared更小，因此小grid只是共同限制，不能解释10.12倍K2差距。原版用约三分之一动态指令、无这项shared额外wavefront开销，并用分工与异步流水隐藏加载；候选则在每chunk串行准备普通shared、重排到UMMA、等待矩阵结果、完成标量epilogue。这些源码与profile证据共同支持数据流设计是主要待改方向，但尚无单变量实验将10.12倍差距按原因百分比分解，也不能据此断言tcgen05无法加速真实KDA。

## 下一轮只保留两个有依据的实验方向

1. **统一持久状态/U的物理布局和寄存器epilogue映射。** 目标是让状态更新与下一次UMMA读取都使用适配的布局，减少冲突及普通shared→swizzle复制。先保持当前同步/输入路径不变，要求同38例、冻结长链仍exact；同时观察实际与理想shared wavefront差距、动态指令和K2时间。只降低shared容量、或只让一处store无冲突而完整链不快，不算成功。
2. **恢复输入/输出的TMA流水与gate共享复用。** 用独立变体保持已验证数值边界，至少把beta/gate和当前/下一chunk工作区的搬运从所有consumer的标量路径移走。下一chunk的输入预取可与当前计算重叠；下一chunk的状态投影必须等待当前BF16状态提交。完整K2资格、缓冲生命周期与tail/varlen必须重新验证，之后比较标量load数、long-score采样与K2/full-forward时间。

两项分开验证，不继续在本轮堆叠更多参数。跨CTA拆value维可以另做吞吐研究，但必须考虑重复K1因子读取、输出/状态分区和每链递推依赖；当前同grid对照没有证据要求先扩大CTA数量。本轮的明确结论是语义可保持、朴素tcgen05移植仍失败，尚未证明存在可战胜原版的配置。
