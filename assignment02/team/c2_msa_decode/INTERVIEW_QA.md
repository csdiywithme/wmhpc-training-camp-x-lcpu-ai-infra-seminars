# C2 专业面试连环追问：MSA sparse decode 的效率、实现与取舍

整理日期：2026-09-14；实测日期：2026-09-13。本文包含 **11 条追问链、54 个主问题**，每条由任务解释推进到效率、源码、质疑与下一步。回答按约30–60秒口述组织。文末另有60/180秒开场、3道白板题、必记数字和证据索引。

阅读重点是**效率、源码与实验**：AI/FLOP/bytes、grid、资源、速度比和计时口径保留深入推算；softmax、scale等数学只掌握正确实现所必需的机制，不要求长篇递推证明。先讲清实际改动与性能证据，再用必要的正确性关系解释边界。

事实说明：本项目在用户授权下由 AI 协作完成代码、实验与报告。下文的“本项目／这次实现”是项目参考答法，不表示本人独立手写了上游实现；面试时应如实说明协作方式与自己能够解释、复现的部分。没有进行真实课堂答辩、上游合并或生产服务集成。

所有性能数字均指已有单卡 **B300 SXM6 AC、148 SM** 实验；TP1/TP4表示单卡head分片形状。最终正收益来自 **PDL=false、16种形状×2个seed的32条记录**：baseline/candidate为1.066–1.268×，几何平均1.117×；不能写成服务端到端收益。

## 链一：你到底优化了什么，与普通 Attention 有什么区别？

### Q1. 用一句话说，这个项目解决什么问题？

本项目研究固定版本vLLM中MiniMax M3的分块稀疏decode attention：每个query和KV head只访问选中的少量KV页。先测小batch的实际开销，再保持partial计算不变，把合并各split结果的merge改成单warp、按输出通道分块。目标是减少一个真实算子链的延迟，而不是重新训练稀疏模型或实现整个推理引擎。

### Q2. “稀疏”体现在哪里？你实现了选择哪些token的算法吗？

输入已经包含top-k逻辑页索引，选择工作由上游indexer完成，本项目没有实现或测量indexer。固定设置是top-k16、每页128个token，长序列下每个KV head最多访问2048个token；再由block table把逻辑页映射到物理KV页。它对选中且因果可见的集合做softmax，不能说与读取全部8192个token的dense attention数学等价。

### Q3. GQA16是什么意思？为什么它会改变decode的矩阵形状？

GQA16表示16个query heads共享一个KV head。本题这些query heads对同一个query位置还共享该KV head的选页集合，因此加载一页K/V后可供16个query heads使用。虽然每个head只有一个decode query，但把组内heads作为M维，QK就形成`[16,128]×[128,128]`的矩阵乘。TP1是64Q/4KV，TP4是16Q/1KV；TP4仍然是GQA16，不是GQA4。

### Q4. 这和FlashAttention的区别是什么？稀疏就不需要online softmax了吗？

两者解决的层次不同。FlashAttention通过分块、在线归一化和片上复用，避免把完整score/probability矩阵写到HBM；它本身并不要求删除attention连接。这里先按稀疏模式限定KV集合，集合内部仍使用分块矩阵乘和online softmax，并因decode并行度不足进一步split-K。稀疏减少访问范围，online softmax减少中间物化，两者可以同时使用。

### Q5. 所以可以宣称你做了完整vLLM优化或多卡TP优化吗？

还不可以。运行的是从固定vLLM快照抽出的独立kernel入口，TP只是每卡head形状，没有真实多卡collective。计时不包含投影、indexer、调度、KV分配、网络和服务排队。能准确说的是“基于固定vLLM实现扩展基准、改良merge并验证条件收益”；模型吞吐、TTFT和TPOT需要接入引擎后重新测。

**面试官在检验什么：**能否划清算法语义、执行优化和系统范围，解释GQA复用从何而来。

**容易答错：**把top-k16当16个token，把TP4当四张卡实测，把FlashAttention说成一种天然稀疏近似。

证据：[原partial入口](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/vllm_msa_ref/sparse_attn.py:234)、[主报告的实验口径](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/WRITEUP.md:11)。

## 链二：既然读KV，为什么不能直接说“HBM带宽受限”？

### Q6. 白板上怎么估算一次decode的主要FLOP和算术强度？

设有效长度L、head dimension D、GQA组大小G，每个query/KV-head组的QK与PV各需要约`2GLD` FLOP，合计`4GLD`。K/V各读一遍时payload为`2LDs` bytes，s是存储字节数，因此KV-only AI为`2G/s`。本题G16、D128、L2048，单组是16,777,216 FLOP；BF16的AI是16 FLOP/B，FP8为32。这里FMA算2 FLOP，暂未计exp、scale和索引。

### Q7. 这个AI为什么不是完整实现的真实AI？遗漏了哪些数据？

它只算理想KV复用。以TP1 B1为例，KV为BF16的4 MiB或FP8的2 MiB，Q和output各16 KiB；只加入一次Q读和O写，AI已降到15.876/31.508。实际还有各split重复读Q、页索引、scale、partial/LSE workspace和可能的spill。baseline workspace容量为`R·Hq·S·(2D+4)` bytes，至少有一次写和merge读；候选分feature后还会重复少量LSE读取。

### Q8. 逻辑bytes和NCU的DRAM bytes不一致，是谁算错了？

二者测的不是同一件事。逻辑bytes是程序需要访问的数据量模型；DRAM counter统计测量窗口中实际到达HBM的流量，受L2命中、事务粒度、writeback时机和重放条件影响。hot graph重复固定地址，NCU则用cache-control=all冷重放，所以不能拿一边的bytes除另一边的时间。某次DRAM write测成0，也不能推出没有partial store，只能说窗口里没有观察到对应HBM写回。

### Q9. roofline给的是时间上界还是下界？什么情况下会误判？

理想roofline给吞吐上界`P≤min(P_compute, BW·AI)`；相应时间下界是`T≥max(F/P_compute, bytes/BW)`，不是“真实耗时至多这么多”。小grid、依赖延迟、非矩阵指令和启动成本会使实际更慢。还要选对计算峰值：当前Triton的FP8 KV先反量化成BF16，SASS用BF16 HMMA，所以不能拿native FP8峰值作它的计算roof。逻辑流量模型也不能冒充已经测出的HBM带宽需求。

### Q10. 你有哪些证据能说明小batch和大batch的限制不同？

TP4 B1 partial只有16个CTA，而B300有148 SM；冷NCU的DRAM read peak约1.847%、tensor elapsed约0.360%，明显不是全卡HBM饱和。TP1 B16 BF16的DRAM read peak增至44.010%，内存压力更突出。FP8在同一代表形状中DRAM读从约67.404 MB降到33.865 MB，partial却从19.968增至27.712 μs。结合格式转换、shared wavefront和spill，说明“字节减半就快一倍”不能成立；这些仍不是各因素的耗时占比消融。

**面试官在检验什么：**能否把计算量、逻辑访存、物理流量、并行覆盖和测量窗口分开。

**容易答错：**说roofline预测时间上界；看到低tensor利用率就认定Tensor Core无用；把低batch和大batch统一归为HBM饱和。

证据：[公式与资源推导](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/analysis/C2_THEORETICAL_ANALYSIS.md)、[完整基线profile](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/experiments/BASELINE_PROFILE.md)、[代表NCU原始目录](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/experiments/results/profile-matrix-b300-20260913T020433Z)。

## 链三：为什么split，怎样保证合并仍是同一个softmax？

### Q11. baseline有多少split？为什么不固定16？

它想把partial grid扩到约256个CTA，又不超过可用top-k页数。令R为flattened query数，普通decode时R=B，代码先算`t=max(1,min(topk,floor(256/(R·Hkv))))`，再向下取2的幂得到S。TP1的B1/4/8/16对应S16/16/8/4；TP4对应全16。partial grid为`(R·S,Hkv)`。因此这是形状启发式，不保证总能达到256，也不表示256是B300的SM数。

### Q12. 在线softmax的(m,l,a)各自保存什么，为什么不用存完整概率矩阵？

m保存目前最大logit，l保存归一化分母，a保存加权V的分子。逐页处理时，只需维护这三个状态；出现更大的最大值，就把旧分母和分子一起重缩放，再加上新页贡献，最后相除得到输出。这样不必把完整概率矩阵写到HBM。需要记住的是分子、分母要用同一尺度，以及空状态必须guard；源码使用exp2和log2-LSE，也要保持底数一致，不必在面试中展开整套递推证明。

### Q13. 为什么不能把各split的输出平均？LSE合并只需记住什么？

各split已经分别归一化，但它们在全局softmax里的权重未必相等，所以还要保存FP32 log2-LSE。只需记住这个权重式：`w_c=2^(ell_c-ell_max)/Σ_d2^(ell_d-ell_max)`，再用这些权重加权各份BF16局部输出。例如两份局部输出是0和1，第二份的指数质量远大于第一份，正确结果就接近1，不能简单平均成0.5。改变split数或归约顺序也可能改变舍入，仍须验收。

### Q14. 空split的LSE是负无穷，怎么避免0乘NaN？全padding呢？

活跃行可能只有一个有效页，却仍按静态形状启动16个split。空split跳过循环，末尾选择归一化scale为0，存LSE为负无穷、局部输出为0；活跃行至少有一个非空split，因而空split的merge权重为0。这个guard避免把`exp(-inf-(-inf))`带入有效结果。全padding行连全局分母也为0，空集合softmax没有定义；接口允许其输出为NaN或零，但必须用active mask排除，并保证不污染旁边的活跃行。

### Q15. 既然(m,l,a)可结合，为什么不直接跨CTA融合掉merge？

数学上可以：把两组的m取max，再按新m缩放并相加l、a。工程上要解决跨CTA同步和存储生命周期。单CTA处理全部页会牺牲split并行；cluster可以提供共驻留和DSM协作，但要另外设计phase、远端数据可见性与buffer重用。普通grid中的自旋或mbarrier不会让尚未调度的CTA自动共驻留。当前只证明merge局部改动有收益，没有实现或排除cluster attention。[CUDA cluster语义](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#thread-block-clusters)

**面试官在检验什么：**是否真正理解softmax状态而非背API，能否把结合性与硬件同步条件区分开。

**容易答错：**平均局部output；用`exp(ell)`合并log2-LSE；把全padding零输出约定当成数学定义。

证据：[split选择](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/vllm_msa_ref/sparse_attn.py:647)、[online与空split处理](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/vllm_msa_ref/sparse_attn.py:312)、[原merge](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/vllm_msa_ref/sparse_attn.py:420)。

## 链四：为什么最终只改merge？代码与计数器怎样对应？

### Q16. 最终代码到底改了哪些内容？是否偷偷改了精度或split？

最终默认保持原partial、split选择、4 warps/3 stages、BF16 partial和FP32 LSE，只换merge。`_page_decode_kernel`与原partial经重命名后AST一致，最终verify、paired和profile的候选源码哈希也相同。仓库还保留子页partial等探索接口，但没有启用为默认策略。因此收益不能归因于减少选中KV或换更宽松的数值路径；merge归约布局变化仍需验收。

### Q17. 单warp、feature tile的映射是什么？会多出多少CTA？

原merge每个query/head一个CTA，处理`[S,D]`。新grid增加feature tile维度：`(R,Hq,ceil(D/TILE_D))`；每个CTA读自己的D区间，并沿S归约。选中策略是S≤8时TILE_D128，否则64，均1 warp。TP1 B1的S16使merge从64增至128个CTA；TP1 B16的S4保持1024个CTA。一个warp并不表示只算32个元素，而是编译器将整个逻辑tile分给32个lane。

### Q18. 多一维grid是不是纯赚？为什么S大时要把D切小？

它有取舍。较小feature tile能减少一个warp要持有的值和沿split归约的工作，但增加CTA和重复LSE读取；太小还会增加启动/调度负担。当前策略来自seed0上的merge配置扫描，随后在独立seed上验证，不能说从理论上证明了全域最优。S大时每个feature对应更多partial，所以把D缩小是一种限制每CTA工作量的办法，不是固定“warp数越少越快”。

### Q19. 69,888→640这个数字是什么？是字节数还是bank conflict数？

它是所选shared-memory wavefront指标的实际计数，不是字节，也不是直接的冲突地址个数。TP1 B1原merge实际/理想值为69,888/8,448，候选为640/640；B16为577,536/86,016→0/0，两个点都消除了excess wavefront。NCU把内存请求分解成可处理的wavefront，实际与理想的差用于观察额外工作；该差不能直接换算成整个kernel多耗了多少微秒。还要结合指令、布局和整链计时。[NCU指标说明](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-reference)

### Q20. 既然merge计数器更好，为什么还要测整条链？

因为partial占比、kernel间隙和PDL依赖会改变总效果。把独立测得的merge时间从chain时间中相减，只能作粗略参照，不能严格满足Amdahl可加性；更不能把跨任务冷NCU的kernel时间当正式速度比。本次收益依据同卡、同输入、同PDL的完整partial+merge graph。PDL打开后一些形状反而退化，正好说明更轻的局部kernel不等于所有执行环境中的整链更快。

**面试官在检验什么：**是否知道真正启用的代码、grid与数据重复代价，是否能正确解释profiler指标。

**容易答错：**把实验接口说成默认优化；称“shared流量降到640字节”；用独立kernel差值证明服务收益。

证据：[feature merge](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/candidate.py:468)、[固定策略](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/candidate.py:685)、[候选完整结果](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/experiments/CANDIDATE_EXPERIMENTS.md)、[最终NCU CSV](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/results/candidate-profile-20260913T022426Z/selected.csv)。

## 链五：TMA不是规则张量搬运吗，怎么处理两级页表？

### Q21. TMA能不能替你做top-k→block table的指针追踪？

这个实现不能让tensor map自动解释两张任意索引表。先由SM读`logical=topk[head,row,slot]`，再读`physical=block_table[request,logical]`，得到物理页号后，将它作为TMA的动态坐标。TMA负责已经定位到的规则页内数据。SASS同时出现两次普通LDG和UTMALDG.4D，证明的是查表与bulk搬运的分工，而不是TMA消除了两次依赖读取。

### Q22. tensor map具体怎么描述这页？为什么K/V拼在最后一维？

缓存逻辑布局是`[physical_page,Hkv,128,256]`，最后256维拼接128维K和128维V。tensor map按最快维在前描述为`[256,128,Hkv,num_pages]`，box为`[256,128,1,1]`，坐标为`[0,0,head,physical_page]`。于是一个CTA搬一整个head的物理页，1/2-byte元素分别是32/64 KiB。这个布局是微实验固定选择，不是声称所有分页KV都能直接用同一descriptor。

### Q23. tensor map有OOB保护，还需要causal mask和无效top-k保护吗？

需要，而且保护位置不同。无效top-k尾槽可能是负poison，必须在读block table前按有效前缀长度跳过；不能先追地址再指望TMA救回来。尾页中的未来token虽然物理地址合法，却在逻辑上不可见，需要attention计算时mask。当前copy实验只搬合法完整页，未实现这两个attention边界；物理OOB机制不能表达`logical_token≤query_pos`的语义。

### Q24. 如果给TMA打开swizzle，就能自动解决shared bank conflict吗？

不能。swizzle改变数据进入shared时的布局，消费端也必须按同一映射读取；具体box、对齐和swizzle span还要符合descriptor规则。本次copy显式使用`CU_TENSOR_MAP_SWIZZLE_NONE`，没有得到swizzle消除attention冲突的证据。真实集成时应按QK/PV消费者布局设计并对比counter，而不是给现有指针公式换一个swizzle枚举。[CUDA TMA swizzle说明](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html)

### Q25. 异步生命周期怎么保证？copy加速能否直接带入attention？

代码先初始化shared barrier并处理async proxy可见性，issuer登记期待字节数后发TMA，消费者等对应phase完成再读shared。这里“数据ready”不等于“buffer可复用”：还要等消费者读完，若以后接异步MMA，还需等待其消费完成。proxy fence提供排序，不能替代完成等待。多stage必须分别管理phase、事务和消费期。本次32配置逐字节通过，copy约1.09–1.61×，但不含attention、scale和流水线，也未与最优cp.async比较。[CUDA异步copy及完成机制](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html)

**面试官在检验什么：**是否理解地址生成与搬运是两个阶段，能否说明物理保护、逻辑mask和异步生命周期。

**容易答错：**声称TMA硬件追两级页表；把swizzle当透明开关；用一次`__syncthreads()`替代异步完成等待。

证据：[paged_copy.cu](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/experiments/paged_copy.cu:23)、[descriptor与NONE swizzle](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/experiments/paged_copy.cu:118)、[TMA实验](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/experiments/TMA_EXPERIMENT.md)。

## 链六：FP8的scale可以移到哪里，为什么代数正确还会验收失败？

### Q26. 当前FP8 KV是直接用FP8 Tensor Core算的吗？

不是。基线加载FP8后转成Q dtype，本题为BF16；有scale时乘FP32 scale再舍回BF16，省略scale时直接cast。QK和PV使用BF16 dot、FP32累加；PV前还把probability转为BF16，这是另一个舍入点。实际SASS有BF16 HMMA及额外FP8 unpack、F16/BF16转换和乘scale。因此FP8节省存储，不保证native FP8算力提升；FP8 B≥4的一些编译配置还出现n_spills=2。

### Q27. 全局K scale能否吸收到Q或softmax scale？V scale呢？

精确算术下可以。若K乘同一个a，logits为`αqᵀ(aK)=(αa)qᵀK`，所以a可合入Q或softmax scale，但必须在max、exp和LSE之前生效。若V都乘同一个b，输出为`b·ΣpV`，b可以放到PV前或最终输出后。两种scale不能混为一谈：K scale改变attention分布，V scale只线性缩放值的加权和。

### Q28. 逐token scale也这样移动吗？为什么V scale不能改softmax分母？

逐token K scale `a_j`对应每一列logit，不能合成一个全局常数。逐token V scale `b_j`可放进`Σ p_j b_j V_j`的分子，但分母仍是原logits的指数和；若把`p_j b_j`重新归一化，就变成另一种分布。这里scale表按KV head和物理token索引，地址是`physical_page·128+offset`，不是逻辑页或top-k槽；页重排时必须连同scale一致更新。

### Q29. 既然数学上可移，为什么你没有把它移出去省乘法？

因为当前实现的舍入边界也是对照语义的一部分。原路径计算`BF16(FP32(BF16(fp8))*scale)`后再做dot；把scale挪到FP32 logits或输出，会交换乘法与BF16舍入，结果未必相同。PV前的`p.to(v.dtype)`又引入概率舍入；gold不模拟这一步，而是检验它等中间近似产生的误差。本次保留partial只改merge；若另做scale融合，应明确是否改变表示输入和数值路径，再独立验收。

### Q30. 怎么设计用例，才能证明scale没有接错？

只用单页、单head和0.25/0.5常数scale太弱。本次加入0.37/0.63、变化的token/head scale、stride=2 backing和poison，再做物理页与scale一致置换。默认gold先得到`BF16(FP32(BF16(fp8))*FP32_scale)`的有效KV，再升FP64计算attention；否则可能把输入表示误差当kernel错误。`raw_scale_gold=True`直接用FP64乘scale，属于另一诊断，不参加冻结门槛。不能只看输出finite或用单一常数证明scale接对。

**面试官在检验什么：**能否同时跟踪代数位置、物理索引和有限精度舍入，识别“随机常数测试通过”的盲点。

**容易答错：**FP8存储等于FP8 MMA；逐token V scale放进重新归一化的softmax；物理页变了但scale不动。

证据：[反量化与dot顺序](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/vllm_msa_ref/sparse_attn.py:339)、[scale接口及支持域](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/validation/ACCEPTANCE.md:7)、[FP8 profile](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/experiments/BASELINE_PROFILE.md)。

## 链七：PDL究竟保证什么，为什么打开以后会退化？

### Q31. PDL和普通CUDA stream顺序有什么不同？

通常同一stream的后继kernel要等前一个完成。Programmatic Dependent Launch允许满足条件的后继提前启动，先做不依赖前驱结果的工作；真正使用结果前仍要同步依赖。它给的是重叠机会，不保证一定并发，更不是取消数据依赖。B300支持这个特性，但“硬件支持”与“这个kernel组合会受益”是两回事。[NVIDIA PDL说明](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html)

### Q32. 在Triton里只传`launch_pdl=True`就足够了吗？

要同时检查launch和kernel内部依赖语义。本题wrapper根据`current_platform.is_arch_support_pdl()`设置`USE_PDL` constexpr，并在支持时传`launch_pdl=True`。内部用`tl.extra.cuda.gdc_wait()`等待前置kernel完成及其写入可见，用`gdc_launch_dependents()`提示后继可以尽早启动。课程shim默认关闭；实验适配明确覆盖该平台判断，候选再传相同的PDL状态。不能只改benchmark标签而不改真实launch。[Triton 3.6.0官方示例](https://github.com/triton-lang/triton/blob/v3.6.0/python/tutorials/11-programmatic-dependent-launch.py)

### Q33. 你把trigger放在partial输出store之前，这不是数据竞争吗？

trigger不是“结果已经写完”的承诺，只是在所有相关program调用或完成后给后继提前launch的资格；后继的wait才等待前置grid完成并保证结果可见。本题partial先wait再读输入，循环结束后trigger，随后归一化并存partial/LSE。merge入口先wait，接着trigger自己的后继，再读partial并归约。这条依赖链不能把trigger误当release数据标志，也不能为了“重叠更多”删掉wait。[gdc_wait](https://triton-lang.org/main/python-api/generated/triton.language.extra.cuda.gdc_wait.html)、[gdc_launch_dependents](https://triton-lang.org/main/python-api/generated/triton.language.extra.cuda.gdc_launch_dependents.html)

### Q34. merge入口立即wait，究竟还有什么可以重叠？

源码里确实没有大量显式的独立算术放在wait之前，所以不能声称当前merge把一大段计算与partial隐藏了。仍可能提前处理launch/prologue或减少kernel间隙，但有多少收益取决于实际调度和编译。候选也沿用wait→trigger→load/reduce/store的顺序，feature分块改变了grid和资源需求。要说明这怎样影响重叠，应采集依赖graph时间线，而不是从CTA更多就直接断言“抢SM导致退化”。

### Q35. 既然PDL=true有退化，你的优化是不是只赢了一个弱基线？

最终结论必须限定。课程harness默认PDL=false，在这个域上16种形状、两个seed都获益；另外已用相同PDL的两端做true对照，TP1 B1和TP4 B4的两种KV dtype均出现退化，整体seed速度比范围0.724–1.388×。因此可以交付条件改良，不能无条件替换生产路径。PDL退化点目前缺完整时间线，尚未证实成因，也没有根据heldout负结果再挑一个dispatch掩盖它。

**面试官在检验什么：**是否理解launch资格、grid完成和内存可见性，是否如实处理生产配置中的负结果。

**容易答错：**wait只等trigger；trigger保证之前store已经可见；PDL一定并发；PDL打开后的差异全归因于CTA争抢。

证据：[partial依赖位置](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/vllm_msa_ref/sparse_attn.py:292)、[trigger在store之前](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/vllm_msa_ref/sparse_attn.py:388)、[merge依赖](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/candidate.py:478)、[真实PDL注入](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/experiments/baseline_workload.py:62)。

## 链八：微秒级加速怎样测，面试官怎样质疑你的统计？

### Q36. 你说11.580→9.131微秒，具体计了什么？

这是TP1 B8 BF16、PDL=false下完整partial+merge算子链的hot CUDA Graph时间。先完成编译和workspace分配，图中重复同一链64次，事件包围一次replay，再除以64。不是单个merge时间，也不含Python wrapper分配、Q投影、indexer或服务请求生命周期。这个代表点按两个seed各自的时间中位数取均值；原始样本和聚合口径都保留。

### Q37. 为什么要预分配、随机交错，还要用独立seed？

预分配让两端比较的是同类kernel工作，避免Python和allocator掩盖几微秒差别。每条记录21对样本，每次随机决定baseline和candidate先后，减轻固定顺序叠加温度、时钟漂移的偏差；每轮两端各执行一次，记录索引仍可配对。调参用seed0，最终性能用101/307，策略在复测前固定。随机交错不能彻底消除所有噪声，所以还要公开样本、慢形状和未锁频事实。

### Q38. 1.066–1.268×、几何平均1.117×，分别怎么计算？

每条seed/形状记录的speedup是baseline样本中位数除以candidate样本中位数。PDL=false有16种形状×2seed=32个ratio，范围为1.066–1.268，几何平均为`exp(mean(log ratio))=1.117`；不是16个跨seed聚合ratio的极值，也不是把全部微秒混在一起取比。速度比r对应延迟下降`1−1/r`，所以最大约21.2%，不能把1.268×写成延迟下降26.8%。

### Q39. 固定地址graph是否只是在测L2？为什么不报一个更真实的API时间？

固定地址会让缓存有复用机会，但并不保证所有KV都在L2。这个实验回答稳定图重放下kernel链的差异；更真实的服务需要变化输入、KV占用、调度和多请求干扰。可另外报告API时间，但要说明它包含什么，不能把它与预分配kernel计时混成一个指标。NCU的冷重放则用于机制诊断，二者互补；采样P95也不是服务尾延迟p95。

### Q40. eager正确就能保证graph/PDL正确吗？你怎样防止“测得快但输出错”？

不能只验eager。本次最终paired在实际计时replay结束后检查两端输出，检查放在计时区间外；64条性能记录共有128份graph后输出，全部finite，最大全局NRMSE约0.002930。这里用固定性能形状的独立参考作补充检查，不替代完整边界suite。它也没有改变metadata来覆盖每个生产Graph更新场景，后者仍需接入测试。

**面试官在检验什么：**是否能还原测量窗口、对照公平性和统计单位，是否检验真正被计时的执行模式。

**容易答错：**32个seed记录叫32种形状；速度提高26.8%等于延迟降低26.8%；把图后sanity check说成生产全域验收。

证据：[图捕获与成对计时](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/experiments/candidate_workload.py:139)、[最终64条原始记录](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/results/candidate-paired-20260913T022747Z/paired.json)、[聚合报告](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/experiments/CANDIDATE_EXPERIMENTS.md)。

## 链九：168条PASS为什么可信？怎么防止参考实现和候选一起错？

### Q41. 你的gold有没有复用同一个split/online softmax？

没有。gold显式按请求、KV head、有效top-k前缀收集K/V，用逻辑位置裁去未来token，再以FP64计算QK、减最大值、自然指数softmax和PV。它不调用baseline、候选或SDPA，也不复用split merge。FP8输入先按基线约定得到有效BF16 K/V，再升FP64；因此测的是对同一表示输入的计算误差，不包含未量化模型输入与量化输入之间的质量差异。

### Q42. 阈值是不是看到候选误差后临时放宽的？

不是。协议先定义指标、floor/cap和seeds；只让baseline用11/29校准，按输入族冻结阈值，再让baseline和候选跑101/307。比如maxabs阈值为`min(0.02,max(0.002,2b+0.0005))`，b是该族baseline校准最大误差。manifest记录协议、spec和源码哈希，验证时重算digest。不能拿“float大概就这样”临时放宽；如果baseline heldout失败，应记录前置失败并分析协议，而不是给candidate发PASS。

### Q43. 为什么不只用torch.allclose或全局相对误差？168条具体代表什么？

全局L2会稀释少数坏行，纯相对误差在接近零时又不稳定。这里同时检查maxabs、逐query/head NRMSE、全局相对L2、逐元素门槛和finite。逐行NRMSE分母是`max(reference_RMS,0.01)`，门槛由预定公式冻结。最终是168条baseline校准、168条heldout baseline和168条heldout candidate，不是168个性能形状。候选最大绝对误差0.00926627，最大行NRMSE0.00397399；通过仅限冻结域。

### Q44. metamorphic test比再加随机seed强在哪里？

它验证应保持不变的语义关系。反转有效top-k前缀改变归约顺序但不改变集合；同时置换物理页、block table和scale应保留结果；在合法已分配的未选中页或未来token放NaN，不能污染可见输出。每个变体先独立算gold，确认不变量成立，再测kernel。浮点归约顺序可能变，所以kernel间不要求bitwise。共享prefix会影响污染范围，本生成器使用request-private pages，不能把同一poison逻辑直接套任意生产共享页。

### Q45. 哪种用例最容易揭露causal、padding或scale的假正确？

强causal probe设置Q=0、所有V=0、只把请求最后一个token的V设成8：早query必须输出0，最后query才可见该值，泄漏会很显眼。padding要混着活跃行测，防止只测全空时忽略污染；短序列能制造空split。scale则用非2幂且随物理token/head变化的表。仍未覆盖非法活跃索引直接越界、共享prefix、其他head ratio/dtype、完整生产PDL边界或sanitizer；PASS不是这些域的默认证明。

**面试官在检验什么：**是否建立独立且可冻结的判定，而不是增加一堆同源随机对拍；是否能解释验证的支持域。

**容易答错：**FP64 gold意味着量化无误差；168条是168种不同shape；全局L2小便认为每行都对；padding输出零意味着空softmax有定义。

证据：[冻结协议](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/validation/ACCEPTANCE.md)、[独立suite](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/validation/suite.py)、[最终heldout](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/results/candidate-verify-20260913T022150Z/candidate-heldout.json)、[内部审查与当前状态](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/validation/INTERNAL_REVIEW.md)。

## 链十：CUTLASS为什么没有统一在B16交叉，FP8误差如何定位？

### Q46. 你比较的是哪个CUTLASS？为什么要固定这么多版本？

入口来自vLLM快照`d4da0c5`，其外部MSA依赖固定为`087c161...`，MSA里的CUTLASS固定为`eb61c911...`。本次构建实际decode planner、kernel和reduction，没有重写它们；仅对未用的prefill adapter做显式拒绝stub，并扩展运行器和对照。否则安装环境里的“最新版CUTLASS”可能不是生产快照真正调用的实现，性能、布局和精度结论都无法追溯。

### Q47. 源码都写了cross16，为什么还要测？静态guard检查什么？

注释和`_MIN_CUTLASS_BATCH_SIZE=16`是这份快照的策略。静态支持还要求选中CUTLASS后端、CUDA的100架构family、支持的FP8 KV、page128/topk16，以及head几何范围；metadata条件还限制dql1–32和总query-head rows≤65536。wrapper本身固定D128。这些限制表示能否走路径，不证明每个被允许的形状更快。本次B<16强制底层调用只做实验，不修改或冒充生产guard支持。

### Q48. 实测交叉点是多少？attention-only和full分别包含什么？

在已测离散点中，TP1 attention-only首次获益是B8，包含本实验Q量化与GPU metadata更新的full graph到B16才获益；TP4分别到B32/B64。attention包含planner选中的forward和可能的reduction；full使用热plan，排除冷编译、CPU prepare和服务调度，普通PyTorch多步量化也不是生产融合量化的最低成本。每行是同卡对照，但小/大矩阵来自两次任务，不能拼成单一同卡曲线，也没测尽整数batch以确定精确交点。

### Q49. 两端精度不一样，怎么知道差异不是你把Q、layout或scale接错了？

Triton是BF16 Q、FP8 KV；CUTLASS额外把Q变成FP8，所以先同时比较原BF16 Q的数学参考和有效FP8 Q的参考，隔离Q量化影响。本实验Q/K/V scales为0.25/0.25/0.5，有效反量化可精确表示为BF16。对齐有效Q后仍有约2.6%–2.8% NRMSE，而Triton约0.3%。再核固定源码和SASS，发现CUTLASS把exp后的probability numerator转成输入Element，即E4M3，因此继续做独立控制，而不是先改阈值。

### Q50. 独立P-FP8控制怎么做？能证明唯一误差来源吗？

用重复同一物理页的诊断输入弱化在线最大值与页调度差异，分别计算数学FP64参考和“exp分子舍入到FP8、分母保留未舍入指数和”的独立参考。NRMSE从0.026060降到0.001681；Q=0时指数分子全为精确可表示的1，误差为0.001663。结合固定源码的转换和SASS，这强烈支持P舍入是主要额外误差，不能证明穷尽所有来源。重复页是专项诊断，不是冻结无重复top-k域；CUTLASS没有跑完整冻结gate，不能声称同精度替代验收通过。

**面试官在检验什么：**是否分清上游策略与新测证据，能否隔离表示误差、算法误差和适配错误。

**容易答错：**把CUTLASS对照称为自己的attention实现；full叫服务端到端；“cross16”当普遍定律；P控制接近就说全部误差来源已形式证明。

证据：[固定依赖运行器](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/experiments/modal_cutlass.py:14)、[生产静态guard](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/vllm_msa_ref/msa_cutlass_sparse_decode.py:207)、[专项全部结果与源码摘录](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/experiments/CUTLASS_EXPERIMENTS.md)、[独立概率控制](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/experiments/results/cutlass-b300-20260913T022105Z/probability-controls.json)。

## 链十一：如果明天交给推理框架团队，你会怎样继续？

### Q51. 现在能提交什么，哪些内容还不能交付为生产功能？

能提交固定候选、独立验收、原始样本、NCU/SASS、完整报告和可复现运行器，说明非PDL课程域的条件收益。不能宣称完成vLLM服务接入、普遍dispatch、PDL问题修复、TMA attention、共享prefix或模型级精度评估。最有价值的交付不是把条件删掉，而是让接手的人知道哪份代码、哪些输入和哪种执行方式得到过什么证据。

### Q52. 第一版生产dispatch会怎么设计？你已经实现了吗？

尚未实现。我会先列出dtype、D、GQA、page/topk、dql、stride、scale类型和PDL的支持合同，并给不满足条件的输入保留原版fallback。性能policy只在独立验证的子域启用；PDL开启默认保留原路径，等完成专门测量后再决定。若想根据当前退化点增加shape例外，也必须当新设计，用新的独立数据验证，不能拿同一heldout结果既调策略又证明泛化。

### Q53. 生产接入后先加哪些测试和指标？

首先验证动态seq_lens、图内metadata更新、地址复用、真实共享prefix、请求混合与padding的正确性，并用sanitizer检查合法支持域中的内存/同步问题。随后把kernel链、Q量化、indexer和engine调度分别计时，再测同负载下的TPOT、TTFT、吞吐和尾延迟。单kernel收益占服务关键路径多少要实际测；不能把这里1.117×乘到模型tokens/s上。

### Q54. 下一轮最值得做什么实验，怎样避免继续盲扫参数？

先补PDL退化点的时间线和依赖边，比较原/新merge的kernel间隙、launch配置和实际重叠；再用受控改动分离feature CTA数与warp布局的作用。FP8则针对转换、shared布局和spill做局部消融，判断绝对微秒收益是否值得改变partial。TMA或cluster属于更大的实现变化，必须先给出支持域、同步设计和端到端算子收益目标。现在这些是下一步方案，不是已经取得的成果。

**面试官在检验什么：**是否能把研究原型转成明确的接口、fallback和验证计划，是否知道哪些证据仍欠缺。

**容易答错：**根据同一heldout再调dispatch然后继续称heldout；把kernel收益外推为服务吞吐；把“计划接入”写成已上线。

证据：[最终适用边界](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/WRITEUP.md:238)、[候选设计时间顺序](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/experiments/CANDIDATE_DESIGN.md)、[最终独立审查](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/FINAL_REVIEW.md)。

## 60秒开场参考

这个项目研究固定vLLM版本的MiniMax M3分块稀疏decode。小batch时，虽然GQA16能复用KV并使用Tensor Core，但工作量和split/merge组织限制了整卡效率。我先在B300完成16种形状的基线测量和NCU分析，再保持原partial不变，把merge改成单warp、按输出通道分块。PDL关闭时，16种形状、两个独立seed的速度比为1.066–1.268×，几何平均1.117×；候选通过168条冻结heldout验收。项目同时保留了PDL开启后的退化和CUTLASS精度/交叉点对照，所以结论是条件算子优化，没有宣称服务端到端加速。

## 180秒开场参考

本项目的起点是一段真实推理算子：固定vLLM快照里的MiniMax M3 MSA decode。它用top-k逻辑页和block table访问物理KV页；本题每页128个token、top-k16、GQA16、D128。重点不是重新选择稀疏token，而是让已选集合上的attention更快且语义可验证。

我先解释并测量工作量。GQA16让一个KV head供16个query heads复用，因此即使decode每head只有一行，QK也能组成矩阵乘；KV-only AI在BF16和FP8下是16和32 FLOP/B。但实际profile显示最小形状只有16个partial CTA，远少于148个SM，不能简单归为HBM饱和；FP8读量减半却更慢，说明转换和片上数据流也重要。

最终采用一个范围可控的改动：保留原partial、split和FP8舍入路径，只把merge改成单warp的输出通道分块。NCU两个代表点的额外shared wavefront消失；正式性能用同卡、同输入、预分配workspace的CUDA Graph随机交错对照，避免用冷profile时间冒充延迟。PDL关闭时，16形状、两个独立seed均获益，几何平均1.117×；TP1 B8 BF16从11.580降到9.131微秒。

正确性先用baseline-only校准冻结阈值，再做独立seed验证。gold直接聚齐合法KV并用FP64计算attention，覆盖变长、尾页、padding、物理页重排和变化的FP8 scale，最终168条candidate heldout通过。性能graph重放后的真实输出也另行检查。

最后我保留了两类会改变工程决策的反例：PDL开启后四组形状退化，不能无条件替换生产路径；CUTLASS并非都在B16交叉，其额外FP8 probability舍入也有独立控制证据。下一步是PDL时间线、生产支持合同和引擎接入，而不是把算子微基准外推为模型吞吐。

## 三道白板题及答案

### 白板题一：TP1 B1的FLOP、workspace和grid

**题目：**给定Hq64、Hkv4、G16、D128、L2048、top-k16，分别算QK+PV FLOP、BF16/FP8 KV字节、baseline workspace容量、partial/原merge/新merge CTA数；指出哪些不是HBM实测量。

**答案：**

\[
F=4H_qLD=67{,}108{,}864,\qquad B_{KV}=2H_{kv}LDs.
\]

BF16为4,194,304 B=4 MiB，FP8为2,097,152 B=2 MiB。S=16；workspace容量`Hq·S·(2D+4)=266,240 B=260 KiB`，其中partial是BF16、LSE是FP32。baseline读写各一次workspace的逻辑下限是520 KiB，不能当DRAM counter；split重复Q读取和其他数据还未加入。partial为`1·4·16=64` CTA，原merge64 CTA；新merge取Dtile64，变成128 CTA，并重复读取两份LSE。它没有改变attention主FLOP或减少选中KV。

### 白板题二：逻辑AI与实测流量——FP8省一半字节，为什么仍慢？

**题目：**从白板题一的TP1 B1推到B16，主QK+PV共1,073,741,824 FLOP，理想BF16/FP8 KV字节为64/32 MiB。实际冷NCU的partial分别为：BF16读67,404,032 B、耗时19.968 μs；FP8读33,865,472 B、耗时27.712 μs。计算理想AI及同测量窗口中的DRAM读取吞吐；为什么不能把这些冷NCU字节除以另一份hot graph时间？

**答案：**理想KV-only AI仍是16/32 FLOP/B，增加batch没有增加单请求的KV复用程度。用同一窗口的`DRAM_read_bytes/time`计算，读取吞吐分别约3.376/1.222 TB/s。FP8读量约减半，耗时却是BF16的1.388倍；因此要继续检查反量化、shared布局、spill及并行覆盖，不能只按HBM字节预测收益。

逻辑bytes、DRAM counter和时间窗口要配套。L2可能服务部分逻辑访问；hot graph又重复固定地址，没有该窗口的流量counter，就不能用冷NCU字节算它的物理DRAM带宽。若用逻辑KV bytes除以hot时间，只能标为逻辑有效带宽。这里的FLOP还只包含主矩阵乘，`F/DRAM_read_bytes`最多是相应的读取口径比值，不能冒充包含所有操作和读写流量的完整AI。

### 白板题三：给TMA画地址、mask和buffer生命周期

**题目：**当前query绝对位置为129，top-k有效前缀包含逻辑页0、1；block table将它们映射到物理页7、3。画出访问及mask，说明页1的scale地址，并解释何时可复用shared buffer。

**答案：**SM读取逻辑页号，再查block table；对页1发TMA坐标`[0,0,head,3]`。逻辑页1的token位置为128–255，仅offset0、1满足`position≤129`，其余即使物理地址已分配也必须causal mask；scale地址为`[head,3·128+offset]`，不能用逻辑页1。无效尾槽在查表前跳过。TMA登记expected bytes后发起，等待对应barrier phase完成后消费者才可读shared；复用还要等消费者消费完，包括未来若使用异步MMA时其读取shared的完成。proxy fence用于跨proxy排序，不等同于TMA完成等待。当前实验单次搬完整页、无attention消费者，所以只验证该有限生命周期。

## 必记数字：记条件，别只背峰值

|项目|需要记住的数值与口径|
|---|---|
|硬件与版本|B300、148 SM、CC10.3；Torch2.10.0+cu130、Triton3.6.0、NCU2025.4.1|
|主矩阵|B1/4/8/16×TP1/4×BF16/FP8 KV，共16形状；seq8192、dql1、G16、D128、page128、topk16|
|理论AI|BF16/FP8 KV-only为16/32 FLOP/B；TP1 B1主FLOP约67.1M，KV为4/2 MiB|
|最小形状覆盖|TP4 B1 partial仅16 CTA；冷NCU DRAM read peak1.847%、tensor elapsed0.360%|
|新merge|S≤8用Dtile128，否则64；1 warp；原partial不变|
|最终性能|PDL=false的32条seed记录1.066–1.268×，几何平均1.117×；延迟下降6.2%–21.2%|
|代表时间|TP1 B8 BF16、PDL=false：11.580→9.131 μs，两个seed中位时间的均值|
|PDL反例|TP1 B1、TP4 B4的两KV dtype退化；PDL=true全体seed ratio范围0.724–1.388×|
|样本量|16形状×2seed×2PDL=64记录；每条21对，共1344对；每graph64次链调用|
|验收|168条baseline校准；heldout baseline168+candidate168 PASS；图后另查128份输出|
|候选误差|full maxabs0.00926627、最大行NRMSE0.00397399；行分母floor0.01|
|wavefront证据|TP1 B1实际/理想69888/8448→640/640；B16为577536/86016→0/0；不是bytes|
|TMA|32配置逐字节正确，copy约1.09–1.61×；无attention/scale/causal集成|
|CUTLASS交叉|已测离散首个获益点：TP1 attention/full B8/B16；TP4 B32/B64|
|P控制|数学gold NRMSE0.026060→独立P-FP8参考0.001681；Q0控制0.001663|

完整pin不要求面试背诵，但需知道如何查：vLLM `d4da0c55af3aa231b6209bf77871f3ed36eab0d2`；MSA `087c161814d4d9c735b46c21212a09e5f8eb92fa`；CUTLASS `eb61c911471867a5fd2466bfd8f29306cea6ebf8`。候选SHA256为`3e4a2ff87b352c7398ce814cf7ea81cf4e4c1fea33fbb5fe6bec8ee115644773`。

## 高风险说法与可用替换

|避免这样说|准确答法|
|---|---|
|“实现了FlashAttention/完整vLLM服务”|“基于固定MSA实现改良merge，独立测量partial+merge链”|
|“优化后所有条件都更快”|“PDL关闭时全主矩阵获益；开启有明确退化”|
|“FP8 Tensor Core自然快一倍”|“当前KV先反量化到BF16 dot；另有转换和布局成本”|
|“把16个split输出平均就对”|“按各自softmax质量/LSE加权”|
|“wavefront下降等于HBM字节下降”|“shared请求的处理工作量指标，需结合其他counter”|
|“TMA替我自动追页表并处理causal”|“SM定位物理页，TMA规则搬运；逻辑mask仍属attention”|
|“PDL trigger后结果已经可读”|“trigger给提前launch机会，后继wait才等待依赖完成与可见性”|
|“168条通过，说明生产都正确”|“通过冻结输入域；Graph/PDL性能样例另有补充检查”|
|“CUTLASS B16是理论阈值”|“上游静态策略；本次实测随head分片与准备成本变化”|
|“消除了2.6%的CUTLASS误差”|“独立P舍入参考解释主要误差来源，没有修改CUTLASS或放宽gate”|
|“实现速度提高26.8%，延迟降26.8%”|“最大1.268×，对应延迟约降21.2%”|
|“准备好的工程方案已经上线”|“给出下一步接入/dispatch计划，目前未实现服务集成”|

## 可当场打开的证据索引

|被追问的主题|直接证据|能证明什么|
|---|---|---|
|任务和完整技术报告|[TASK](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/TASK.md)、[WRITEUP](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/WRITEUP.md)|范围、六讨论点、最终限定结论|
|数学与算术|[理论报告](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/analysis/C2_THEORETICAL_ANALYSIS.md)、[CPU checks](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/analysis/theory_checks.py)|FLOP/bytes/split、数学反例；不是GPU计数器|
|实际原版代码|[sparse_attn.py](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/vllm_msa_ref/sparse_attn.py)|partial、log2-LSE、scale、PDL与原wrapper|
|最终候选|[candidate.py](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/candidate.py)、[设计记录](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/experiments/CANDIDATE_DESIGN.md)|启用的merge与未选探索的区别、先测再设计|
|原版profile|[BASELINE_PROFILE](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/experiments/BASELINE_PROFILE.md)|16形状与7代表输入NCU、SASS、资源和测量边界|
|最终成对性能|[paired.json](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/results/candidate-paired-20260913T022747Z/paired.json)、[测量代码](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/experiments/candidate_workload.py:156)|21对样本、顺序、两seed、PDL与graph后检查|
|最终机制|[candidate NCU CSV](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/results/candidate-profile-20260913T022426Z/selected.csv)|两个merge代表点实际wavefront计数|
|冻结与heldout|[ACCEPTANCE](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/validation/ACCEPTANCE.md)、[calibration](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/results/candidate-calibrate-20260913T020759Z/calibration.json)、[heldout](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/results/candidate-verify-20260913T022150Z/candidate-heldout.json)|gold、预定阈值、支持域、哈希与逐例状态|
|TMA|[CUDA微实验](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/experiments/paged_copy.cu)、[TMA报告](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/experiments/TMA_EXPERIMENT.md)|真实两级查表与4D搬运，copy范围|
|CUTLASS|[固定依赖runner](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/experiments/modal_cutlass.py)、[专项报告](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/experiments/CUTLASS_EXPERIMENTS.md)|pin、交叉表、计时边界、P控制与SASS|
|交付及独立审查|[FINAL_REVIEW](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/FINAL_REVIEW.md)、[INTERNAL_REVIEW](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/validation/INTERNAL_REVIEW.md)|当前状态、独立核对和课堂替代范围|

官方语义链接已于整理时核验：前文的NVIDIA PDL、CUDA异步copy与NCU文档，以及Triton v3.6.0示例/API。它们用于解释语义；具体实验结论仍以固定源码及2026-09-13原始结果为准。本次整理没有重新启动GPU，没有修改冻结协议或将未来方案写成已实现结果。
