# C1 FlashKDA：专业面试连环追问与口述回答

整理日期：2026-09-14；实验完成日期：2026-09-13。依据题目固定快照、最终 WRITEUP、原始 GPU 结果与源码整理；本次没有重跑 GPU。全文有 **9 条主链、45 个编号问题**，每条链沿上一回答继续追问。按最新偏好，优先讨论efficiency：并行度、FLOP、资源、tile和测量；非效率数学仅解释必要直觉与工程后果。先口述每问的回答段，再按追问补紧随其后的效率公式或数据。

贡献边界只说明一次：课程给定 FlashKDA / FLA 快照、任务和参考；推导、隔离实验、split2 挑战、GPU 运行和材料通过 AI 多代理协作完成，非个人独立手写实现。下面用“本项目”陈述可核验工作；实际面试应按自己能解释和复现的部分说明参与程度，不虚构个人提交、上游合入或生产经历。

证据始终分五层：**解析恒等式 → 隔离数值机制 → GPU 微基准 → 完整 forward → 模型质量**。前一层成立，不能自动替后一层背书。所有本地链接使用原仓库绝对路径，复制本稿后仍可定位证据。

## 60 秒项目介绍

这个项目研究 FlashKDA 在 B300 上为什么仍使用 SM80 世代 MMA。我们先对齐 gate 和状态布局，复现原版对 FLA 的六个形状，再通过 NCU、SASS 和定向数值实验检查瓶颈。发现C16兼顾小矩阵求逆成本、数值范围和warp形状；K2的96个CTA少于148个SM，每条链又有512个chunk，工作覆盖和依赖都值得关注。挑战实现把同一 head 的 value 列分给两个 CTA，正确性逐位通过，但六个形状全部变慢，速度比只有 0.571–0.751。另一方面，tcgen05 在 state update 微基准的增量成本上有优势，完整单轮却仍慢。因此结论是保留原默认路径，继续重构完整 K2；不能把新指令的局部优势直接写成完整算子收益。

## 180 秒项目介绍

这个项目研究FlashKDA在B300上为什么仍使用传统MMA。我的分析重点是：工作怎样分给GPU、哪些成本真正限制执行，以及一个看似提高并行度的改法为什么没有收益。

先看执行结构。FlashKDA分为K1准备和K2状态递推：K1能沿chunk并行，K2一条序列一个head对应一条状态链。T8192、C16时每条链512步；单序列H96只有96个CTA，H12只有12个，都少于B300的148个SM。H96缩到H12后，K1时间明显下降，K2仍接近790微秒。这说明不能只看总FLOP或Tensor Core峰值，还要看独立工作量、链长度与片上数据流。

性能比较前先核对语义。固定版本的bounded gate要对应FLA的safe_gate，状态布局必须一致，还要关闭FLA到FlashKDA的自动dispatch。正式六个H64/H96形状中，原版对同语义FLA为1.48–2.89倍；这是上游实现的复现。实际B300二进制仍出现HMMA，没有自动升级成tcgen05。

挑战实现选择按value列拆CTA，因为不同列不需要相互归约。把原4个compute warp分到两个CTA后，逐位正确性通过，但六个形状全部变慢，baseline除以候选只有0.571–0.751。NCU显示每CTA完整shared仍未缩小、寄存器65升到80，并新增local请求。因此增加CTA的同时，必须重构资源和存储路径；单纯把grid翻倍没有意义。

另一个方向是tcgen05。K2的瘦矩阵经转置能得到合法tile，但TMEM、同步和后续转换有成本。最终微基准在state update的整轮增量上发现1.73倍有利信号，完整单轮却仍慢，所以不能据此宣布完整K2加速。扩大chunk也不是免费办法：求逆工作、shared预算和数值风险都会增加；C16/32/64的求逆需要6/8/10次dense GEMM。

结论是保留当前默认实现，下一步针对shared分配、输入复制和store做单变量消融，或者把完整K2数据流接入tcgen05后再测。项目价值在于固定版本复现、实际候选与可追溯的负结果，发布标准仍是完整forward的正确性和持续收益。

## 追问链一：从算子四步追到chunk并行边界与布局

### Q1. 你说这是线性注意力，具体线性在哪里？它到底计算什么？

**口述回答：**这里是对序列长度线性。每个head维护一个固定大小的状态，token到来后依次做四件事：衰减旧状态、用key读已有预测、把value与预测的差写回、用query读取更新后的状态。它不是把softmax attention原封不动算快，而是一个不同的递推算子。性能优化首先要保持这四步的顺序。

**效率要点：**每head状态有 $D_kD_v$ 个元素，单token主工作是 $O(D_kD_v)$；Dk=Dv=128时是16,384个状态元素。历史长度增大不会增大这个状态矩阵，但会延长递推链。输出包含当前token的更新。[naive逐token实现][naive]。

### Q2. 如果我照这四步写参考，实现还可能在哪里对不上？

**口述回答：**先查输入是不是同一个数学量。FlashKDA接收raw gate和beta logits，内部做激活和q/k归一化；参考可能需要已经激活的值。当前gate限定在负五到零，beta在零到一，输出还有默认scale。其次查状态转置：FlashKDA按value-first存，naive按key-first存，两维都128时错误不会改变shape。

**实现检查：**默认输出scale为 $1/\sqrt{128}$；FlashKDA state为[N,H,V,K]，naive为[N,H,K,V]。用非对称随机初始状态检查转置，不能只测零state；不要把raw gate重复激活。源码先把自然对数gate换到底2，再用EX2，换底本身不改变数学语义。[gate源码][gate]、[接口适配][adapter]。

### Q3. token有依赖，为什么chunk里面还能用矩阵乘？哪些工作能够并行？

**口述回答：**chunk方法没有删除依赖，而是把一块token之间的影响预先整理成系数矩阵。只依赖这块q/k/gate的准备工作，不需要知道入口状态，所以可以提前并行做；等入口state可用后，再用这些系数和value生成输出、更新末状态。这样把很多小矩阵向量操作组织成Tensor Core能处理的矩阵乘。

**依赖图：**当前chunk的q/k/gate → K1准备 → workspace；前一chunk末state + workspace + value → K2 → 当前输出和末state。K1能沿chunk并行，K2的入口state仍串联。完整代数推导放在 [理论报告][theory]，面试主要解释这个并行边界。

### Q4. 既然“预先整理系数”，哪些边界最容易悄悄改掉？

**口述回答：**预测误差读取更新前状态，所以只依赖更早token；输出读取更新后状态，所以包括当前token。对应系数矩阵一个去掉对角线，一个保留对角线，不能统一使用同一mask。beta还是每token的写入强度，不是随便移到任意阶段的公共scale；移动它可能改变不同token之间的影响。

**工程对照：**源码INV是块内逆变换，右端还要乘beta；不要把它误解为已经包含beta的整体操作。tail要保持真实序列边界，padding不能引入额外写入或衰减。先用1个token、非零初始state、短尾块测试，再测整齐大shape。[K2 residual与beta][k2residual]。

### Q5. 为什么分K1/K2，不能一个kernel全融合，省掉workspace和launch吗？

**口述回答：**可以研究融合，但节省workspace流量的同时，可能把原来并行的K1准备串进每条K2状态链。K1负责归一化、激活和块内系数，K2负责依赖入口state的计算；这个拆分是在存储流量与并发之间取舍。只说“少一个launch”不够，要看高并发准备工作是否被迫串行，以及shared和寄存器能否承受融合。

**后续形状讨论用的名字：**K2五类GEMM分别是入口预测 $K_dS$、入口输出 $Q_dS$、残差逆变换、块内输出、末状态更新。其中R表示块内inverse，M表示块内输出系数，U表示本块整理后的更新值；这些符号只用于标识乘法形状。数学转置不等于物理搬运免费，K2现有value-first布局、LDSM/STSM和寄存器片段已经紧密配合。[K1][k1]、[K2布局][k2layout]。

**面试官在检验什么：**能否从更新顺序推到代码分工，而非只记“线性attention很快”。**容易答错：**把 gate logits 当衰减；漏 scale；输出用旧状态；随意移动beta；把相同 shape 当布局一致。

## 追问链二：从求逆工作量追到大chunk的精度与成本

### Q6. Neumann不是要求范数小于一吗？这个疑问为什么会影响chunk选择？

**口述回答：**这里用的是有限展开：严格下三角的C×C矩阵满足 $L^C=0$，不需要依赖无限级数收敛。工程上关键是形成这个inverse要多少工作，以及中间值能否准确保存。当前采用doubling，每层做一次平方和一次扩展乘加，C16/32/64分别需要6/8/10次dense GEMM。

**成本：**对 $C=2^m$，单次dense GEMM按 $2C^3$ FLOP计，求逆总量为 $4(m-1)C^3$。因此每token/head成本为3,072/16,384/81,920 FLOP。次数只从6增到10，但矩阵本身变大，所以不能把C64的逆成本误认为只增加67%。[实际FP16 inverse][inverse]。

### Q7. 作者说inverse元素在负一到一，那半精度怎么还可能出问题？

**口述回答：**这个最终结果的界依赖归一化key、合法beta和不扩张的衰减，但即使承认它，也不能约束具体算法的中间值。doubling会先产生很大的矩阵幂和部分和，最后再相互抵消。最终答案很小，不代表计算途中只需要很小的动态范围或很少有效位。

**与效率的关系：**改成更高精度会增加中间存储和数据搬运，换稳定求解组织又可能引入更多顺序依赖。必须把数值路径和性能路径一起设计。对齐keys的压力例能暴露这种中间增长；只看随机keys误差很小，不能证明更大chunk可安全沿用相同算法。[机制报告][numericreport]。

### Q8. 给个GPU上的具体反例。只是C64才坏吗？

**口述回答：**C16就能出现严重抵消，不必等到Inf。直接构造逆算法输入、beta为0.990234375时，显式FP16 MMA的角点得到负一，参考应几乎为零，最大逆误差达到一。C32/C64的beta为一案例则在P8部分和就溢出。它直接限制“扩大chunk只改常量”的做法，但不能把inverse误差一说成完整KDA输出误差一。

**必记边界：**C32的P8最大幅值475,020，大于FP16最大65,504；P8是累计到七次幂的部分和，不是 $L^8$。beta=1、gate=0是边界压力例，有限实数logits的精确sigmoid只能趋近端点，浮点激活可饱和；beta=0.990234375提供邻近案例。这45个测试直接构造L，跳过完整gate/key前级，也不是模型quality实验。[原始JSON][numericjson]。

### Q9. 全部换FP32会怎样？为什么不能简单以精度换性能？

**口述回答：**只加宽accumulator还可能把阶段结果存回FP16；全部中间矩阵都保留FP32，则存储和搬运增加，也仍没有解决该doubling的最坏抵消。C64、beta为一时，中间量到约 $10^{17}$，最后要消成零或一；系统条件数只有128，GPU最大逆误差却达到2,147,483,648。问题是计算路径，不是所有FP32求解器都不可靠。

**代价口径：**同样C×C中间矩阵从FP16改为FP32，元素存储由 $2C^2$ 变为 $4C^2$ 字节；这只是局部预算，不等于完整kernel时间翻倍。C32同例全FP32最大误差为8。正确方向是比较更稳的求解组织、保存精度和完整数据流成本，而不是仅检查结果finite。[阶段数据][numericreport]。

### Q10. 如果不用doubling，下一步如何评估？为什么不直接CHUNK=64？

**口述回答：**先评估直接单位下三角求解或分块求解，用压力例和残差检验稳定性，再测其依赖和资源成本。它们可能避免显式巨大矩阵幂，但会改变顺序组织和Tensor Core效率。C64同时改变求逆、指数表示和shared容量，所以不是修改一个常量就能得到新指令喜欢的大tile。

**保留的效率公式：**沿当前dense组织、Dk=Dv=D，主GEMM每token/head为

\[
F/C=6D^2+8CD+4(\log_2C-1)C^2.
\]

D128下C16/32/64分别117,760/147,456/245,760 FLOP；shared纸面预算约96/158/306 KiB。大C减少chunk边界次数，却未减少每token的 $6D^2$ 主项；以上是原组织外推，没有完整C32/C64候选性能。[预算推导][theory]。

**面试官在检验什么：**能否同时评估chunk成本增长、中间数值风险与精度/资源取舍。**容易答错：**要求 $\lVert L\rVert<1$；“inverse 有界所以不会溢出”；把 P8 写成 $L^8$；将 FP32 失败泛化为所有三角求解失败。

## 追问链三：从强衰减的指数范围追到弱衰减的状态舍入

### Q11. C16 的范围优势究竟是 BF16 的什么特性？第 18 个 token 为什么会坏？

**口述回答：**这是指数动态范围，不是有效位数。G表示块内累计gate；当前把衰减拆成 $e^{G_i}$ 和 $e^{-G_j}$，最强 gate 为负五，C16 对应正负八十，C32 则到正负一百六十。代码先用带 FTZ 的 FP32 EX2，再转 BF16；在 $G=-90$ 时衰减变零、逆因子变 Inf，所以会出现零乘无穷。BF16 和 FP32 指数范围相近，单纯改成 FP32 不能解决。

保守正规范围要求 $5C<126\ln2\approx87.3365$，还要给近似指数留裕量；17 也满足此纸面界，16 不是唯一数学可行整数。第18步 $G=-90$ 的失效由 [GPU 六个范围案例][numericjson] 确认；FTZ 清除 subnormal 输入/结果的语义见 [PTX EX2 规范](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-ex2)。弱 gate 的 C32/C64 不一定溢出，不能说所有输入都在第18步坏。

### Q12. 那直接算指数差，或者给前缀和减一个中心值，可以免费修好吗？

**口述回答：**直接算恢复项的指数差，可以避免零乘无穷，但很远的影响仍可能下溢。中心化能改善两个因子的范围，不过入口state、末state恢复和下一chunk的尺度必须同步补偿；只给cumsum减常数会改变算法。它还引入状态转换和尺度管理成本，不能只算指数变得finite的好处。

**工程结论：**最强gate下C32可居中到约正负80，C64仍有正负160；状态和值自身的范围也要检查。本项目仅验证因子范围与直接差分，未实现完整rescaled KDA。即使因子不溢出，inverse的中间抵消仍是另一问题。[范围实验边界][numericreport]。

### Q13. 状态不是有遗忘吗，为什么还要特别测BF16误差？

**口述回答：**遗忘可以抑制过去的扰动，但每次保存state都会加入新的舍入误差。如果某个方向衰减非常弱，新误差就能长期留下；如果很多次小变化都低于BF16的舍入门槛，状态还可能根本不动。bounded gate限制最强衰减，并没有让所有方向都快速遗忘。

**验证与性能取舍：**要同时测强衰减、弱衰减和高度相关keys；不能用“精确递推不扩张”代替实际精度验证。改变持久state精度会增加存储和片上带宽需求，因此需要单独消融，而非把所有整实现误差都算到state上。详细误差界仅作为延伸阅读：[理论报告][theory]。

### Q14. 能不能给一个“FP32 FMA也救不了”的舍入数字？

**口述回答：**考虑一部分state没有被当前key更新，只做很弱的衰减。C16每块应该把一乘成0.998401，但BF16保存时又舍入回一；下一块仍从一开始，反复保存就停住。FP32 FMA能提高计算精度，却救不了最后那次低精度保存，这也是弱衰减必须加入测试的原因。

**具体数字：**每token gate为−0.0001，16步的块衰减系数是0.9984012793。BF16在1下方的邻点为0.99609375，中点为0.998046875，所以该系数保存后回到1。隔离模型重复到T65536时真值约0.001424976，BF16保存模型仍为1。这个反例来自 [CPU状态保存检查][cpuchecks]，不是完整GPU或模型quality的实测失效。

### Q15. 那你实际验证了什么？把 state dtype 设成 FP32 能做消融吗？

**口述回答：**实际有三层对照：相同舍入的官方 oracle 检查实现一致性，独立 FLA naive 检查整条实现的数值偏差，同门函数的 FLA chunk 作额外参考。短和变长共30组，另有4组弱衰减长序列测到32768。它们没有证明模型质量，也没有把总误差全部归因于 state。当前 FP32 state 接口仍会在内部转成 BF16，不能用它冒充真正持久 FP32 状态的消融。

30组与4组长序列均对官方舍入 oracle 逐位通过；长序列相对 naive 的最大 output/state RMSE 为 **0.816647%/0.750500%**，最差1024-token输出窗口 **0.823819%**。naive 内部强制 FP32，不是 FP64 gold；不同 T 不保证同一前缀，不能宣称误差随时间单调增加。真正消融要固定其他舍入，只改内部持久 state，并报告性能成本。[长序列数据][longprecision]、[FP32→BF16 转换][statecast]、[每块 BF16 写回][statewrite]。

**面试官在检验什么：**能否区分强衰减范围、弱衰减舍入和模型质量。**容易答错：**把 BF16 的指数范围说成 FP16；“改 FP32 一定解决”；漏 rescale 状态补偿；“小于1%就对模型安全”；把隔离反例说成完整 GPU 结果。

## 追问链四：从 SM80 指令追到 tcgen05 合法 tile 与布局成本

### Q16. 你怎么知道它真的还在用 SM80？会不会编译器自动升级了？

**口述回答：**我不会只根据 C++ 类型名判断。实际 B300 扩展的 sm_103a 二进制反汇编出现 `HMMA.16816.F32.BF16` 和 `HMMA.16816.F16`，没有 HGMMA 或 UTCHMMA；自定义 tcgen05 微基准则有 UTCHMMA。这里 SM80 指指令家族，设备本身仍是 B300，也不能把多个模板的静态指令总数当一次调用的动态次数。

源码中 inverse 明确用 `SM80_16x8x16_F16F16F16F16_TN`，逻辑16×16由两个N8 atom组成；K2 用 BF16 输入和 FP32 累加。[inverse atom][inverse]、[实际 SASS 与 profile 目录][profile96]。

### Q17. tcgen05 最小 M64，C16 是不是只有四分之一利用率？

**口述回答：**对 K1 的孤立16×16输出，直接补到M64确实只有四分之一的有效算术比例；但不能推广到整个K2。还要先限定具体指令族：这里讨论本次双操作数都在shared的SS路径：普通 dense、非 .ws、单CTA、FP16/BF16，不能拿其他稀疏或双CTA规则混着推。

该SS路径允许 $M=64/128$，$N=8\ldots256$ 且步长8，单指令 $K=16$。K128 是多次K16累加，不是一条K128指令。形状利用率为“有效乘积/补齐后乘积”，不是实测 Tensor Core 峰值比例。[PTX 形状表](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-matrix-shape)。

### Q18. 既然不能推广，给我列出 K2 的合法映射。

**口述回答：**关键是把瘦矩阵乘整体转置。16×128的结果转成128×16，M和N就都合法了。这样K2的五类乘法都能找到不增加有效乘积数量的映射；K1的16×16 inverse转置后仍是16×16，所以那个限制还在。

| K2 计算 | 原逻辑 M×N×K | 利用 $(AB)^T=B^TA^T$ 后 |
|---|---:|---:|
| $K_dS_0,Q_dS_0$ | 16×128×128 | 128×16×128 |
| $R\times residual,MU$ | 16×128×16 | 128×16×16 |
| $K_r^TU$ | 128×128×16 | 原形状即可 |

例如残差乘法整体转置后，beta也要对应到正确的列；若仍按原来的方向缩放，就是语义错误。这里保留形状和数据流判断，完整矩阵等价推导见理论报告。

### Q19. 都能合法映射了，为什么不直接替换 MMA 调用？

**口述回答：**合法形状只是第一步。原K2已经把残差、beta、BF16转换、输出和状态更新紧密接在寄存器数据流里；tcgen05的累加结果在TMEM，需要决定谁读取、何时完成、后续操作数放哪里。代数上可以转置，不代表这些中间值能以目标布局零拷贝地继续用。短K、串行阶段里，固定开销尤其可能盖过乘法收益。

例如 $R\times residual\rightarrow U\rightarrow MU$ 中，还夹着逐元素操作和指定舍入点。本次测量是A/B均在shared的SS路径；若改成A在TMEM的TS路径，必须另查其布局与形状约束，不能继承SS的所有自由度。必须画清 shared→MMA→TMEM→寄存器或后续MMA的数据路径、producer/consumer完成条件；不能把 `commit`、`wait`、布局转换删除后再与完整旧路径比。[K2 融合计算][k2residual]、[tcgen 微基准][tilecode]。

### Q20. 如果只给你一周继续做，你会从哪里切进去？

**口述回答：**优先保留K1和C16，围绕K2 state update做完整数据流原型，因为它有128×128的形状，也有微基准有利信号。先把TMEM消费、state衰减和BF16保存全部接起来，再看完整K2、完整forward是否变快。若只看到单块GEMM提升而总时长不动，就要转向依赖和资源问题，不继续堆指令峰值论据。

已有信号：state update整轮增量为SM80 **193.59 ns**、tcgen05 **111.83 ns**，约1.73×；但完整单轮kernel为 **8.632/13.620 μs**，tcgen05仍慢。研发假设因此是“数据流整合后可能受益”，不是“已经完成K2加速”。

**面试官在检验什么：**能否把数学转置、指令合法性、物理布局和端到端收益分开。**容易答错：**所有C16乘法只有25%利用率；K128是一条指令；MMA合法就能直接替换；将tile1.73×称为完整KDA收益。

## 追问链五：从 K2 低并行度追到 affine scan 的代价

### Q21. 你说 K2 并行度低，用多少个独立工作单元来解释？

**口述回答：**原K2一条序列一个head对应一个CTA，CTA遍历整条chunk链。单序列H96时只有96个CTA，H12时只有12个，都少于B300的148个SM。T8192、C16意味着每条链512步，增加T只延长链，不增加独立链数量。这个映射比单独报FLOP更直接地解释覆盖不足。

\[
\text{grid}_{K2}=N_{seq}H_{local},\quad
H_{local}=96/TP,\quad \text{chain length}=\lceil T_{seq}/16\rceil.
\]

这里只用每卡head形状研究TP8的H12情形，没有进行真实跨卡通信实验。`waves/SM=0.32` 是NCU按可驻留CTA容量定义的量，不能直接把它当“32%的SM工作”。

### Q22. 官方八条1024明显更快，那把8192直接切成八段不就行了？

**口述回答：**官方那八条是真实独立序列，每条有自己的入口状态。把一个8192序列切成八段并把状态清零，会改变后七段的历史。保持语义的切分必须先求各段入口状态，这正是原本的依赖。可以换并行算法求入口，但不能用重置状态免费获得更多CTA。

每一段的入口必须是前一段的真实末状态。变长还会出现长尾：六条不等长链比六条等长链更难均衡；总T相同不代表执行时间应相同。原始三种序列组织见 [官方H96数据][official96]。

### Q23. 那状态矩阵能不能沿任意维度切给多个CTA？

**口述回答：**沿value列切最自然：所有列用同一套key/gate系数，但不同列之间没有求和归约。沿key维切则不同，预测需要把所有key维的贡献加起来，切开后就必须通信或归约，不能各自独立更新再拼接。这个区别直接决定切分后的额外开销。

**效率判断：**value split可以增加CTA、减少每CTA负责的列数；成本是重复读公共输入和可能重复分配buffer。key split除了这些，还要考虑归约、同步和中间存储。本项目选择前者，首先验证无跨列依赖，然后用实际资源与时间判断收益。

### Q24. 不切value，多个head塞进一个CTA，或者persistent queue呢？

**口述回答：**多head进CTA可能交错独立链来掩盖局部等待，但会减少CTA数，低NH时未必改善全卡覆盖；简单把不同head的矩阵拼接，还可能产生不应有的跨head乘积。persistent queue可改善变长任务分配，但当前K2已经驻留整条链，队列不会创造额外链。两者都需要针对不同并行域分析。

同样，“2-CTA”不能混成一个概念：本项目split2是两个独立CTA计算不同value列，**不是**实现了tcgen05的CTA-group协作、cluster同步或分布式TMEM。后者有额外协议，不能用split2负结果直接否定。

### Q25. 能不能用parallel scan，把chunk间的串行依赖也解决？

**口述回答：**这是合法研究方向：状态递推可以表示为affine变换，精确算术下变换复合有结合性。但scan减少依赖深度的同时，可能显著增加工作。这里不是两个标量相加，而是矩阵变换；复合后矩阵可能变稠密，低秩形式的秩也可能增长，还要恢复每块的正确入口state。

**效率核心：**原逐token主工作是 $O(D^2)$，一般稠密变换复合达到 $O(D^3)$，并增加中间存储。D128、C16时要把新工作、存储和并行收益列全；浮点重关联还可能改变舍入，需要重新验收数值。本项目没有完整scan候选，也不会把“存在结合性”说成已经消除了成本。[并行度分析][theory]。

**面试官在检验什么：**能否识别真正独立的维度，以及区分work与span。**容易答错：**增加T就增加CTA；把长序列切段重置；key/value切分等价；多head拼接无交叉项；scan具有结合性就一定更快。

## 追问链六：从 split2 数学正确追到资源负收益与消融

### Q26. 你实际实现了哪种并行重构？不要只讲想法。

**口述回答：**生成器在远端复制的FlashKDA树里构建独立模块，把原4个compute warp分到两个CTA，每个处理64个value列、保留2个compute warp。K1和每列算术顺序不变，写回只覆盖自己负责的输出和状态范围，包括varlen尾块。第一版保留完整shared分配和输入TMA，所以它是可验证的具体实现，也带着已知冗余。

源码通过 `blockIdx.z` 选择列分区，store warp等到输出可读后合作写所属列，再释放buffer。没有跨CTA写同一元素，也没有跨value归约。[challenge_build.py][challengecode]；上游快照保持不变，可分别核对 [生成patch][challengepatch] 和 [挑战运行结果][challengedir]。

### Q27. 你怎么证明切分没有破坏状态、尾块和FP32接口？

**口述回答：**先有按列独立的解析证明，再做实际输出和末状态的逐位比较。候选在30组小形状precision案例、8种状态接口组合以及3个H96完整长度形状上全部与baseline逐位一致。baseline又对官方舍入oracle和独立naive验证，因此对照链可追溯。逐位一致的价值是说明这次切分保留了已有数值路径，不代表baseline本身对精确递推零误差。

8种接口组合为“是否有初始state × 是否请求最终state × BF16/FP32外部state dtype”。尾块只写actual_len和本分区列，不能仅检查大整齐矩阵。证据：[小形状precision][precision]、[大形状正确性][largecorrect]。

### Q28. 正确性过了，性能究竟怎样？有没有挑最好的一行？

**口述回答：**六行全部报告，全部负收益。H12覆盖TP后低head情形，H96覆盖原题主形状；每组都有单长序列、不等长六序列和八条独立短序列。baseline除以候选只有0.571到0.751，所以候选比原版慢。这是对该具体资源和store设计的否定，不能包装成“优化成功”，也不能证明其他value切分必然失败。

| H | 序列组织，总T8192 | baseline ms | split2 ms | baseline/split2 |
|---:|---|---:|---:|---:|
| 12 | 1×8192 | 0.8370 | 1.4058 | 0.595× |
| 12 | 不等长6序列 | 0.3568 | 0.4751 | 0.751× |
| 12 | 8×1024 | 0.1604 | 0.2293 | 0.699× |
| 96 | 1×8192 | 1.0761 | 1.8648 | 0.577× |
| 96 | 不等长6序列 | 0.8861 | 1.4493 | 0.611× |
| 96 | 8×1024 | 0.7092 | 1.2423 | 0.571× |

20 warmup、100 iterations×5 repeats、普通event计时，未锁频；不把这组顺序测量称为跨卡交错复验。[全部样本][challengetiming]。

### Q29. CTA翻倍却更慢，能证明就是spill造成的吗？

**口述回答：**能证明资源和流量发生变化，不能证明全部损失只来自spill。H96单序列K2 grid从96到192，但每CTA shared仍是98,432字节，寄存器从65到80，并出现baseline没有的local请求。读流量也增加，store路径也改变。多个因素一起变，必须做消融才能分配因果贡献。

NCU K2时间 **793.440→1,583.040 μs**；local load/store请求约 **1,572,864/1,180,416**，baseline为零；DRAM读 **0.886→1.106 GB**，tensor elapsed **18.27%→9.16%**。65是baseline实测每线程寄存器数，分配粒度按72计，不能混写。[详细对照][experiments]。

### Q30. 失败以后下一步怎么排实验？“再调调参数”不算答案。

**口述回答：**先分别改三个有证据的成本：将完整shared state缩成列分块实际需要的范围；研究输入读取和stage是否能减少；单独比较store组织。每步保持相同正确性域，记录寄存器、local请求和完整fwd时间。只有单因素版本拿到数据，才组合有利改动，避免一次改完后不知道为什么变快或变慢。

预注册观察：shared缩小是否增加驻留空间、local请求是否下降、负收益在H12/H96是否一致、varlen尾块是否仍正确。输入中的公共矩阵不一定能够免费跨CTA共享；若引入cluster协作，必须把同步和布局代价重新计入。以上是下一步方案，尚无这些消融的GPU结果。

**面试官在检验什么：**能否从实现细节解释负结果并提出可证伪的下一步。**容易答错：**只报CTA翻倍；不交代完整shared仍在；把相关counter当唯一原因；将8种state接口误说成真正FP32持久状态验证。

## 追问链七：从微基准数字追到同步正确性与可比性

### Q31. 你的微基准究竟模拟了什么，没模拟什么？

**口述回答：**它模拟shared输入固定、FP32依赖累加、最后写回一次输出的重复GEMM。四种形状来自转置后的K2乘法、state update和补齐的小tile，并测试不同CTA数。这比单纯测一条指令接近某些依赖模式，但没有完整KDA的gate、beta、U布局、持久状态和输入流水，所以它只能回答特定数据流下的局部问题。

配置为2种实现×4种形状×4种CTA数（1/12/96/148）×2种内部轮数（1/64）=64配置。两边最后应得到 `loops × A Bᵀ`；输入为可精确核对的BF16数，累加FP32。M64案例两侧都做完整M64，不能把它当成原M16 inverse与padded版本的完整公平比较。[最终源码][tilecode]。

### Q32. 同一个矩阵反复乘，编译器会不会把循环消掉？SM80每轮都重新读shared吗？

**口述回答：**循环使用依赖累加，正确结果随loops变化，同时检查实际SASS，不能只相信源码里写了循环。但即使乘法没有消失，编译器仍可能把不变的操作数保留在寄存器。当前SM80路径就有这种复用，所以不能声称每轮都承担相同LDSM流量；这也是不能拿它去证明新旧shared加载布局谁绝对更好的原因。

验证不只检查第0个CTA：先用NaN填充全部输出，运行后逐元素检查finite，再与所有CTA的gold比较。最终64配置全部通过，最大误差0；这是该输入/配置域的执行验证，不是通用布局形式证明。SASS分别确认HMMA/UTCHMMA。[最终结果快照][tiledir]。

### Q33. 1.73倍这个数是怎么算的？为什么完整单轮还是慢？

**口述回答：**每个样本计时20个kernel组成的graph，再重放20次，除以400得到kernel时间。对1轮和64轮分别取5个样本的均值，用差除以63估计增加一整轮工作的边际成本。它摊掉了一些固定开销，所以与完整单轮不是同一个指标。tcgen05的TMEM和同步固定成本会改变小工作量下的结论。

\[
\Delta t=\frac{\operatorname{mean}(t_{64})-\operatorname{mean}(t_1)}{63}.
\]

96 CTA、128×128×16：SM80/tcgen05增量 **193.59/111.83 ns**，比例1.73；完整1轮 **8.632/13.620 μs**，64轮 **20.828/20.665 μs**。增量是固定grid的整轮时间，包含依赖/同步，不是每CTA时间，更不是单条MMA延迟。[最终汇总][tilesummary]。

### Q34. 你提到微基准曾有barrier问题，具体怎么发生，怎么修？

**口述回答：**旧版让所有warp独立等同一个parity，同时issuer继续推进后续轮；某个消费者若落后多个phase，只看一个bit就可能混淆轮次。最终改为只有issuer warp在循环里推进MMA、commit和wait，其他warp等循环结束后的CTA同步再读TMEM。这样消除了该消费者落后的用法；还修正了只看第0个CTA、fmax可能掩盖NaN的验证问题，随后完整重跑。

协议：shared写入及可见性准备→issuer发GEMM→commit登记先前异步MMA完成时的barrier arrival→issuer等当前phase→切换parity→下一轮；循环结束CTA rendezvous→TMEM load→写输出→再次CTA同步→warp集体release/free。commit发射不等于MMA已完成，释放前的同步也不能省略。CuTe底层选举issuer，不是32线程各自发一份相同MMA；TMEM分配/释放按warp集体接口执行。[最终实现][tilephase]。PTX要求parity跟踪当前或紧邻上一phase，不能当无界序号使用：[mbarrier等待规范](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-mbarrier-try-wait)。

### Q35. 修好以后你敢把这组结果说成tcgen05更适合state update吗？

**口述回答：**我会说它提供了一个值得继续验证的有利点。最终源hash和结果一一对应，64配置、320样本、全部CTA检查已经过关，旧目录保留但不用于结论。不过输入固定、每轮commit/wait、SM80操作数复用和未锁频都是明确边界。要升级成“适合完整K2”，还需要接上实际数据流后的端到端实验。

最终可信目录时间戳 **022453478641Z**，源码SHA256为 `878273ce2a4f5b4a4129374a49b45833f592daf2eb9e005557061c9057367e41`。瘦形状128×16×16的增量是 **23.25/83.48 ns**，tcgen05更慢，不能只展示state update一行。[完整四形状数据][tilesummary]。

**面试官在检验什么：**能否审计自己的benchmark，包括发现并修正历史错误。**容易答错：**跳过finite检查；把parity当无限轮次；把SM80源码copy当动态LDSM证据；把固定grid增量当指令峰值；继续引用修复前结果。

## 追问链八：从同语义 benchmark 追到 NCU 的证据强度

### Q36. 官方benchmark已经给了，为什么还要做“同语义”对齐？

**口述回答：**能运行不代表算同一个东西。固定版本FlashKDA使用bounded sigmoid gate，而原benchmark没有给当前FLA版本传safe_gate。我们先保存原样结果，再只补safe_gate和现行state_v_first参数，对齐后才计算同算子加速比。还关闭FLA到FlashKDA的dispatch，否则可能变成被测实现和自己比较。

固定 `FLA_FLASH_KDA=0`；不把FLA GDN列计作同KDA语义；FP32-state列仅是接口精度，不代表不同内部状态算法。初始`arange`大状态用于复现官方性能，精度验证另用随机状态。速度比的分子分母、gate、dtype、布局和公共API范围必须一起交代。[最终WRITEUP][writeup]。

### Q37. 给出可复现的环境和正式数字，别只说“B300更快”。

**口述回答：**实验是2026年9月13日单B300 SXM6 AC、148SM、CC10.3；FlashKDA固定1ce47ea，FLA固定a3edffc完整版本，构建sm_103a。官方六行D128、总T8192同语义对照显示原版对FLA为1.48到2.89倍。它说明该固定实现和输入域上的复现收益，不能拿另一台GB200表格反推架构因果加速。

| H | 序列 | FlashKDA ms | 同gate FLA ms | FLA/FlashKDA |
|---:|---|---:|---:|---:|
| 96 | 1×8192 | 1.0789 | 2.0788 | 1.93× |
| 96 | 1300/547/2048/963/271/3063 | 0.8892 | 2.1029 | 2.36× |
| 96 | 8×1024 | 0.7110 | 2.0575 | 2.89× |
| 64 | 1×8192 | 0.9882 | 1.4672 | 1.48× |
| 64 | 同上不等长6序列 | 0.6704 | 1.4839 | 2.21× |
| 64 | 8×1024 | 0.4826 | 1.3869 | 2.87× |

30 warmup、200 iterations×5 repeats、未锁频；driver580.95.05、CUDA toolkit13.1.80、Torch2.10.0+cu130、Triton3.6.0、NCU2025.4.1。FLA pin `a3edffc39eb5a3d45e9deab5ff9ec4f14f88474d`，CUTLASS `5c149f5`。[H96][official96]、[H64][official64]。

### Q38. 看起来FLOP不大，那到底compute-bound还是memory-bound？

**口述回答：**先按K1/K2分开。K1并发高、DRAM和L2活动较高，准备和访存成本值得关注；K2全卡HBM与tensor都没饱和，而且CTA少、依赖链长，更像工作覆盖和链内数据流限制。不能因为没到算力峰值就断言memory-bound，也不能因为用了Tensor Core就称compute-bound。

H96单序列的NCU：K1 grid **49,152**、duration **272.544 μs**、DRAM **4.529 TB/s**、L2 elapsed **72.32%**；K2 grid **96**、duration **793.440 μs**、DRAM **1.353 TB/s**、L2 **25.55%**、tensor elapsed/active **18.27%/28.40%**。这些是带观测条件的counter，需连同shared、寄存器、active warps、等待采样一起解释。[NCU原始报告][ncureport]。

### Q39. 只凭96个CTA就说并行覆盖重要，会不会是你自己的猜测？

**口述回答：**还做了H12对照。head数缩小八倍，K1时间从约273微秒降到42微秒，K2仍约787微秒，接近H96的793微秒；每head的512-chunk链长度没变。这比只看一个低利用率counter更支持并行覆盖与依赖的解释。但它还不能精确分开shared等待、指令依赖和调度的贡献。

H12/H96 K2 NCU时间为 **786.528/793.440 μs**。较少head降低总工作却没有等比例降低最长链延迟，是该映射下的重要证据；把H96变H12也改变了其他运行条件，因此不是所有局部机制的单因素证明。[H12 profile][profile12]。

### Q40. 为什么NCU时间与benchmark对不上？纸面强度模型还能用吗？

**口述回答：**NCU详细采集会replay并按配置处理cache，环境与普通event计时不同；完整接口还可能包含辅助kernel和workspace操作。所以正式速度比用同口径benchmark，NCU用于解释资源和执行现象。纸面强度适合检查数量级和提出假设，但请求字节不是实际DRAM字节，更不是观测到的带宽上限。

当前C16逻辑请求模型约 **38.6 FLOP/B**，忽略部分参数/overfetch/spill等；它无法凭一个roofline标签解释低NH的K2。active与elapsed峰值百分比的分母也不同。要提高证据强度，下一轮应做同卡交错、更多重复和时钟记录；本次C1没有完成跨卡交错复验，不报告服务p95。[成本与计时边界][theory]、[最终实验报告][experiments]。

**面试官在检验什么：**是否先核语义再测速度，能否把counter当证据而非标签。**容易答错：**FLA间接调用自己；把不同gate或GDN混算；GB200/B300跨环境比归因架构；混合NCU与event时间；仅凭一个stall名称定因。

## 追问链九：从研究结论追到发布决策和可交付范围

### Q41. 如果你是作者，v2到底出不出SM100专版？先明确选项。

**口述回答：**当前不会把split2作为默认发布；保留C16/SM80默认路径，继续研究条件启用的SM100后端。现有完整实现的优势、C16范围和小inverse组织是保留理由；K2合法映射和state update局部信号是继续研发理由。这个结论比“新指令没用”更准确，也给出了可以推翻当前决策的实验条件。

本次证据否定的是**已经实现并测量的split2设计**，没有证明原版全局最优，也没有完成真正tcgen05 K2。因此“专版研究”与“候选发布”是两个不同决定。

### Q42. 给后端加个shape dispatch，只在有利形状启用，不就能发布了吗？

**口述回答：**首先得有完整候选在那些形状上有利，当前split2六行没有这样的点；tile微基准不能直接生成完整算子的dispatch条件。未来若出现稳定正收益，再先固定策略、覆盖dtype/state/varlen/tail并在独立输入上验证。维护双后端还要记录硬件、工具链和回退条件，不能只在调参样本上挑赢家。

发布门槛至少包括完整fwd同语义正确性、压力输入数值界、重复测量的净收益、对不支持域回退。不要为了让候选通过而候选后调宽误差门槛；C1本次没有类似C2的完整冻结验收manifest，不能借用别的任务的样本数。

### Q43. 你说正确性与收益都要过，模型质量还需要单独测吗？

**口述回答：**需要。相同舍入oracle逐位通过，只说明实现符合既有数值路径；对naive误差小，也只覆盖所测输入。真实模型会有相关keys、长程记忆和多层传播，不能从单算子RMSE直接推断模型质量。模型级评估要使用真实checkpoint和任务，设定比较对象与指标，这次没有这些数据。

可按四层验收：解析特殊例→定向/随机算子误差→完整模型输出与任务指标→服务吞吐和尾延迟。实际完成到有限域完整forward；模型quality、真实服务、跨GPU TP通信都未测。弱衰减与aligned keys压力例的意义是扩大覆盖，不能冒充真实模型失效证据。

### Q44. 那这项目有什么可交付价值？失败优化是不是等于没做成？

**口述回答：**有价值的不是“失败”这个词，而是把一个假设做成可执行实现，证明它保持语义，量化它为什么没有达到目标，再收窄下一步设计空间。这里交付固定版本、原始计时、SASS、NCU、数值反例、正确性记录和生成器；别人能沿证据复查。还明确发现了微基准验证协议的问题并重跑，避免用错误数据得出架构结论。

复现入口是 [modal_experiments.py][runner]，模式包括 `official/profile/challenge/challenge_profile/precision/long_precision/numeric/tile`。不能把现有脚本文件名当来源证明：最终tile用hash对应快照；numeric工作版与原快照有一次causal-pair诊断舍入差异，复刻原JSON应使用 [实测numeric快照][numericsnapshot]。

### Q45. 简历写“FlashKDA加速2.89倍”会被追问什么？你怎样准确收尾？

**口述回答：**首先会被问分母是谁、改了哪段代码。2.89倍是原版FlashKDA对同gate FLA的复现，不能写成新增实现带来的加速；1.73倍只是tile整轮增量，完整单轮仍慢。准确收尾是：完成上游复现和瓶颈分析，实现并验证value split2，公开其六形状负收益，并给出数值反例和下一步完整K2路线。

一句话版本：**“以同语义基线、机制实验和实际负收益候选，解释FlashKDA在B300上保留传统MMA的合理范围；尚未交付更快的完整C1候选。”** 若被问明天做什么，回到Q30的单因素消融和Q20的完整K2数据流原型，而不是重复新硬件峰值。

**面试官在检验什么：**能否做技术决策、尊重证据边界并清楚交付。**容易答错：**负结果证明所有SM100路线失败；微基准直接生成dispatch；算子RMSE等于模型质量；把上游对FLA收益写成新增代码收益。

## 五道效率白板题：把结论变成可计算的预算

以下练习不另计入45问，全部围绕效率、资源和实验设计。复杂数学证明留在理论报告，不作为本稿的背诵重点。建议各用3–5分钟完成。

### 白板题A：算CTA数量与链长度，解释为什么总T相同时间不同

**题目：**B300有148个SM，K2每序列每head一个CTA。比较H96/H12下的1×8192和8×1024，C16时各有多少CTA、每条链多少chunk？

**答案：**

| 场景 | K2 CTA数 | 每条链chunk数 |
|---|---:|---:|
| H96，1×8192 | 96 | 512 |
| H12，1×8192 | 12 | 512 |
| H96，8×1024 | 768 | 64 |
| H12，8×1024 | 96 | 64 |

总token工作相近，但独立链数和最长链不同；增加T只延长链，增加真实独立序列能增加CTA。不能把同一长序列重置成八条，否则改变历史状态。148个SM也不意味着148个CTA一定达到峰值，还需看资源和链内执行。

### 白板题B：扩大chunk前，列出FLOP和shared预算

**题目：**沿当前dense组织，C16/32/64分别需要6/8/10次C×C inverse GEMM。D128下只说“chunk数少四倍”哪里不对？

**答案：**单次dense GEMM为 $2C^3$ FLOP，除以C得到每token工作；完整主GEMM预算还包括三个C×D×D和其他C×C×D乘法。

| C | inverse FLOP/chunk | inverse FLOP/token | 全部主GEMM FLOP/token | 原buffer shared纸面预算 |
|---:|---:|---:|---:|---:|
| 16 | 49,152 | 3,072 | 117,760 | 约96 KiB |
| 32 | 524,288 | 16,384 | 147,456 | 约158 KiB |
| 64 | 5,242,880 | 81,920 | 245,760 | 约306 KiB |

C64的主GEMM每token约为C16的2.09倍，shared还需要重构；收益只能来自减少边界、改变tile/dataflow等，不能声称计算量少四倍。表不含rescale、新求解、标量操作和spill，不是已实现C32/C64的计时预测。

### 白板题C：合法tile、转置和补齐比例

**题目：**本次SS、普通dense、CG1 BF16路径允许M64/128、N为8的倍数、单指令K16。判断16×16×16、16×128×128、128×128×16三种GEMM。

**答案：**第一种直接pad成64×16×16，有效算术比例16/64=25%；转置也不能改变16×16输出。第二种用乘积转置得到128×16×128，有效乘积不增加，但K128要用8次K16累加。第三种128×128×16原形状即合法。

**判分点：**25%只描述直接补齐后的有效乘积比例，不是tensor峰值；转置后的布局转换可能有成本；本次A/B都在shared，改成TMEM A时必须另查TS约束，不能自动假设零拷贝。

### 白板题D：区分微基准增量、完整kernel与完整算子

**题目：**96CTA的state update，SM80/tcgen05完整1轮为8.632/13.620 μs，64轮为20.828/20.665 μs。如何解释约1.73×，是否能发布完整KDA加速？

**答案：**增量估计为 $(\operatorname{mean}(t_{64})-\operatorname{mean}(t_1))/63$；用完整未舍入样本算得193.59/111.83 ns，比例约1.73。表中四舍五入后的时间手算会有小差异。这是固定grid多一整轮工作的边际时间，含同步与依赖，不是单条MMA延迟。

完整1轮仍是tcgen05更慢，64轮也只是接近；更没有包含gate、U转换和真实state流水，所以不能发布完整KDA收益。可将1/64两点拟合成“固定成本＋每轮增量”模型，估计交点约62轮，但没有测量全部轮数，不能当精确阈值或生产dispatch条件。下一步应连接完整K2，再报告K2与完整forward两个层级。

### 白板题E：从split2计数器提出单变量消融

**题目：**CTA从96到192，shared每CTA仍98,432B，regs65到80，local请求从零增加，K2时间793.440到1,583.040 μs。下一轮怎么做，如何证明原因？

**答案：**先固定输入、gate、正确性和计时协议，按顺序设计独立候选：只缩shared分配；只改变输入/阶段搬运；只改变store组织。每个记录shared、regs、local请求、实际流量和完整forward时间；单项有利后再组合，避免只测一个“大改版”。

若local请求下降但forward不快，就不能把spill称为全部原因；若shared缩小却仍覆盖不足，要看grid、链长和真实驻留。输入重复和store也可能各有成本，不能仅凭两个counter相关就分摊百分比。每一步仍要回归H12/H96、varlen/tail和state接口。

## 必记数字与容易串错的口径

| 必记项 | 数字 | 必须连带说出的边界 |
|---|---|---|
| 主形状 | C16、D128、T8192、H96；512 chunk/head | 单长序列只有96条独立链；H12不是实测跨卡TP |
| B300资源 | 148 SM，CC10.3 | 单卡、固定版本、未锁频 |
| 上游复现 | H64/H96六行，同gate1.48–2.89× | 分母是FLA；非新增代码收益 |
| split2 | 六行0.571–0.751× | baseline/candidate，小于1是变慢 |
| split2资源 | shared98,432 B保持；regs65→80 | baseline分配粒度72；多个因素未单独消融 |
| K2链证据 | H12/H96约786.528/793.440 μs | NCU观测，不能替代完整fwd正式计时 |
| 指数范围 | $126\ln2\approx87.3365$，g−5第18步 | 指独立因子、FTZ路径；弱gate不一定失效 |
| Neumann | C16/32/64：6/8/10 GEMM | 有限幂零恒等式，不要求范数<1 |
| FP16结构反例 | C16 beta0.990234375逆误差1 | 直接构造L的机制测试，非完整输出误差 |
| FP32结构反例 | C64 beta1误差2.147e9，κ∞128 | doubling稳定性问题，非系统极端病态 |
| tile | state update增量1.73×；单轮8.632/13.620 μs | 固定grid边际成本；不是完整K2收益 |
| tile验证 | 64配置、320样本、all-CTA最大误差0 | 仅最终022453目录；输入域有限 |
| 完整精度 | 30短/varlen＋4长；T最大32768 | naive内部FP32；没有模型quality |
| 长序列误差 | output0.816647%、state0.750500% | 整实现相对naive；不全归因BF16 state |

避免三种“偷换”：**接口FP32→持久状态FP32；inverse最终有界→中间算法稳定；局部性能优势→完整算子或模型收益。** 未完成项目包括完整C32/C64 KDA、完整tcgen05 K2、实际FP32持久state消融、模型质量、真实服务和跨卡复验。

## 原始证据与复现索引

| 要核实的内容 | 入口 |
|---|---|
| 题目、最终结论、讲述材料 | [TASK][task]、[WRITEUP][writeup]、[DEFENSE][defense] |
| 递推、chunk与布局 | [naive][naive]、[K1][k1]、[K2][k2layout]、[FLA适配][adapter] |
| 范围/成本/稳定性推导 | [理论报告][theory]、[CPU检查][cpuchecks] |
| C16/32/64隔离数值机制 | [说明][numericreport]、[JSON][numericjson]、[实测快照][numericsnapshot] |
| H96/H64正式同语义复现 | [H96 JSON][official96]、[H64 JSON][official64] |
| 指令与资源profile | [H96原始NCU][ncureport]、[H96目录][profile96]、[H12目录][profile12] |
| split2具体变更与负收益 | [生成器][challengecode]、[原始计时][challengetiming]、[资源说明][experiments] |
| 正确性与长序列 | [小形状][precision]、[H96完整形状][largecorrect]、[长序列][longprecision] |
| 最终tile结果与同步实现 | [源码][tilecode]、[结果目录][tiledir]、[汇总JSON][tilesummary] |
| 命令、版本与退出码 | [Modal入口][runner]，各结果目录的commands/environment等原始记录 |

复现采用WRITEUP中的固定版本和 `uv ... modal==1.5.5 python -m modal run` 命令；数值脚本可独立运行。当前numeric工作版仅有causal-pair诊断的乘积舍入调整，原JSON保留旧口径，该字段不能直接当实际K1单项误差；45个inverse和范围结论未受此调整影响。检查来源时以实测快照SHA与原始数据对应关系为准，不能虚构个人commit。

[task]: /Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c1_flashkda/TASK.md
[writeup]: /Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c1_flashkda/WRITEUP.md
[defense]: /Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c1_flashkda/DEFENSE.md
[naive]: /Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c1_flashkda/fla_kda_ref/naive.py:51
[adapter]: /Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c1_flashkda/fla_kda_ref/backends/flash_kda.py:60
[gate]: /Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c1_flashkda/FlashKDA/csrc/smxx/fwd_kernel1.cuh:305
[k1]: /Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c1_flashkda/FlashKDA/csrc/smxx/fwd_kernel1.cuh:340
[k2layout]: /Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c1_flashkda/FlashKDA/csrc/smxx/fwd_kernel2.cuh:72
[k2residual]: /Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c1_flashkda/FlashKDA/csrc/smxx/fwd_kernel2.cuh:586
[inverse]: /Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c1_flashkda/FlashKDA/csrc/smxx/utils.cuh:189
[statecast]: /Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c1_flashkda/FlashKDA/csrc/smxx/fwd_kernel2.cuh:299
[statewrite]: /Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c1_flashkda/FlashKDA/csrc/smxx/fwd_kernel2.cuh:718
[theory]: /Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c1_flashkda/analysis/C1_THEORETICAL_ANALYSIS.md
[cpuchecks]: /Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c1_flashkda/analysis/theory_checks.py
[numericreport]: /Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c1_flashkda/analysis/C1_NUMERIC_EXPERIMENTS.md
[numericjson]: /Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c1_flashkda/results/c1-numeric-h96-20260913T021503565747Z/chunk-numeric.json
[numericsnapshot]: /Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c1_flashkda/results/c1-numeric-h96-20260913T021503565747Z/chunk_numeric_gpu.py
[official96]: /Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c1_flashkda/results/c1-official-h96-20260913T020630921340Z/semantic-matched-benchmark.json
[official64]: /Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c1_flashkda/results/c1-official-h64-20260913T022812279519Z/semantic-matched-benchmark.json
[profile96]: /Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c1_flashkda/results/c1-profile-h96-20260913T020442527901Z/
[profile12]: /Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c1_flashkda/results/c1-profile-h12-20260913T020932999579Z/
[ncureport]: /Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c1_flashkda/results/c1-profile-h96-20260913T020442527901Z/baseline.ncu-rep
[challengecode]: /Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c1_flashkda/challenge_build.py
[challengepatch]: /Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c1_flashkda/results/c1-challenge_profile-h96-20260913T021340887396Z/challenge.patch
[challengedir]: /Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c1_flashkda/results/c1-challenge-h96-20260913T020956599379Z/
[challengetiming]: /Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c1_flashkda/results/c1-challenge-h96-20260913T020956599379Z/challenge-timings.json
[precision]: /Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c1_flashkda/results/c1-challenge-h96-20260913T020956599379Z/precision.json
[largecorrect]: /Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c1_flashkda/results/c1-challenge_profile-h96-20260913T021340887396Z/large-shape-correctness.json
[longprecision]: /Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c1_flashkda/results/c1-long_precision-h4-20260913T022617825299Z/precision.json
[experiments]: /Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c1_flashkda/analysis/C1_EXPERIMENTS.md
[tilecode]: /Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c1_flashkda/tile_microbench.cu
[tilephase]: /Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c1_flashkda/tile_microbench.cu:60
[tiledir]: /Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c1_flashkda/results/c1-tile-h96-20260913T022453478641Z/
[tilesummary]: /Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c1_flashkda/results/c1-tile-h96-20260913T022453478641Z/tile-summary.json
[runner]: /Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c1_flashkda/modal_experiments.py
