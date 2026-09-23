# C2 答辩材料：小 batch MSA decode 的收益与边界

本文件是可排练的讲稿与问答准备，不是答辩已举行的记录。真实外组挑战、现场提问和现场答辩尚未发生；本轮只有独立代理审查和保存的 GPU 实验。下列时间为建议分配，口述时展示表格和关键代码，不逐字朗读公式、路径及附录。

**当前交付范围已由用户确认：以答辩材料和内部独立审查记录交付，替代本次工作中的真实外组审查与现场答辩活动。** 这项确认改变的是交付范围，不改变历史事实；没有向其他学员发送消息，也没有声称课堂活动已经举行。冻结验收协议中的“尚未运行”“仅计划审查”等表述保留协议写定时的状态；当前结果以本文证据表、[内部审查记录](validation/INTERNAL_REVIEW.md)及对应原始 JSON 为准，不为更新状态而改变协议哈希。

## 10 分钟中文讲稿

### 0:00–0:45｜问题与结论

我们研究的是 MiniMax M3 的主 attention decode。Indexer 已经给出十六个 KV 块，每块一百二十八个 token，所以每个 query 最多关注两千零四十八个 token。TP1 是六十四个 query heads、四个 KV heads；TP4 是十六个 query heads、一个 KV head，两者 GQA 都是十六。

我们选择挑战 a：先测量原始 Triton，再做有条件的改良，并与固定版本的上游 CUTLASS 对照。结论有三个：小 batch 的瓶颈随形状变化；merge 有可直接测到的布局成本；所谓 CUTLASS 在 batch 十六交叉，必须说明量化、metadata 和精度口径，不能当成硬件定律。

### 0:45–2:00｜先测量，再决定优化方向（讨论点 5）

我们先在 B300 跑完 TP1、TP4，batch 一、四、八、十六，BF16 和 FP8 的十六个基线形状。第一份完整数据在 UTC 02:00 左右落盘，随后才开始候选探索。原始 kernel、编译信息和 Nsight Compute 报告都保存了。

需要分清两个时间口径：热 CUDA Graph 测真实的重复 kernel 链；NCU 清缓存后反复采集硬件计数器。它们不能混在一张性能排名里。比如 TP4、batch 一、BF16 的热链约六微秒；冷 NCU 的 partial 单独就约七点六微秒，这不是矛盾。

TP4、batch 一只有十六个 partial CTA，而 GPU 有一百四十八个 SM。NCU 的 DRAM 读取仅到峰值约百分之一点八五，Tensor Core elapsed 利用率约百分之零点三六，因此不能说它已经吃满 HBM。另一方面，TP1、batch 十六的 BF16 partial，DRAM 读取达到约百分之四十四，瓶颈已经不同。

merge 还有明确的共享内存证据：TP1、batch 一的实际 wavefront 是六万九千八百八十八，理想值只有八千四百四十八。这个差距支持我们检查归约布局，而不是凭感觉宣布“第二次启动太贵”。

### 2:00–3:00｜算术强度与 Tensor Core（讨论点 1）

主矩阵乘包括 QK 和 PV，按一次 FMA 两个 FLOP，总量是 `4 R Hq L D`。同一个 KV head 的十六个 query heads 可以共享 KV 读取，所以只算 KV payload，BF16 的理想算术强度是十六 FLOP 每字节，FP8 是三十二。

TP1、batch 一的两次矩阵乘约六千七百万 FLOP，BF16 KV 四 MiB，FP8 两 MiB。但计入重复 Q 读取、partial 和 LSE 的写回读取，完整请求账目的强度会降到约十三点四和二十三点一。这些是逻辑字节模型，不是实测 HBM 流量。

Tensor Core 有用武之地：时间维只有一个 query token，GQA 仍提供十六行。实际 SASS 也证明基线已使用 BF16 HMMA。因此我们的目标不是“从没有 Tensor Core 变成有”，而是改善供数、归约和并行度。Roofline 给吞吐上界、时间下界，也不能单靠这个下界证明“不值得优化”。

### 3:00–4:05｜两阶段是否应该融合（讨论点 2）

融合当然可以省 partial 往返和一次 launch，但代价是减少 split 并行度。TP4、batch 一若每个 query/KV head 只留一个 CTA，整个 GPU 只剩一个主 attention CTA，可能比当前十六个更差。我们因此没有直接把两阶段硬塞进一个 CTA。

cluster 在数学上可行：每个 split 带自己的归一化输出和 log-sum-exp，可以重新加权合并。但 mbarrier 只负责同步，不负责算归约，也不保证普通 grid 的任意 CTA 共驻留。可移植 cluster 最多八个 CTA，而小 batch 的 split 是十六；还要解决 DSM 生命周期、phase、资源共调度，不能把它当免费全局共享内存。本轮没有实现 cluster，不把理论可行性写成实测收益。

最终候选保持原 partial，修改 merge 的特征分块和 warp 数。S 不超过八时使用一百二十八维 tile，否则用六十四维，每个 CTA 一个 warp。候选 NCU 的两个代表点，merge 的 excessive shared wavefront 都降到零；这是实际机制证据，但最后仍由完整链的性能决定是否值得使用。

### 4:05–5:10｜TMA 能不能访问两级页表（讨论点 3）

问题要拆成两步。Tensor map 描述规则多维张量，不能替线程执行“先读 top-k，再读 block table”的指针追踪。但线程拿到物理页号后，可以把它作为动态坐标，用 TMA 搬该页里的规则数据。

我们按真实 KV 布局做了独立 CUDA 微实验。线程路径与 TMA 路径都执行 global 到 shared 再写回 global，用 CPU 独立构造期望输出。十六种形状、两种路径，共三十二个配置全部逐字节一致。SASS 同时出现两次普通 LDG 和真正的 UTMALDG.4D，正好验证了这条分工。

但这只证明访问机制，并不是 attention 已加速。微实验没有 softmax、反量化和计算流水线，也没比较最优 cp.async 实现。真实接入还必须在查页表前屏蔽无效 slot；TMA 的物理越界语义不能替代 causal mask，更不能保护已经发生的非法页表读取。

### 5:10–6:20｜FP8 scale 应放在哪一层（讨论点 4）

标量 K scale 在精确算术中可以并入 score scale，标量 V scale 可以移到输出端。但不同物理 token 的 scale 不能这样整体移出去；尤其 V scale 必须进入每个 token 的加权求和。scale 的地址是物理页号乘一百二十八再加页内偏移，不能误用逻辑块号。

有限精度还要看舍入位置。原 Triton 把 FP8 解码为 BF16，再做 FP32 scale，相乘后舍回 BF16，然后做 BF16 dot。它没有因为 KV 存成 FP8 就变成原生 FP8 矩阵乘。我们的验收明确以这种有效输入为 gold；另保留直接 FP64 scale 的诊断，二者不偷换。

实测也很说明问题：TP1、batch 十六，FP8 虽然把 DRAM 读取减半，partial 反而比 BF16 更慢。源码和 SASS 给出大量解包与格式转换，NCU 显示更多共享内存 wavefront，部分 FP8 形状还有 local spill。仅说“FP8 带宽省一半，所以应快一倍”是不成立的。

### 6:20–7:45｜为什么不是统一在 batch 十六交叉

我们找到 vLLM 实际依赖的 MSA commit，连 CUTLASS submodule 也固定。两端在同一任务、相同输入上测，Triton 使用 PDL，CUTLASS 同时报 attention-only 和包含 Q 量化、热 metadata 更新的 graph。

在已测离散点里，TP1 的 attention-only 在 batch 八已开始获益，包含准备算子后，首个获益点变成十六。TP4 在十六仍更慢：Triton 约十二点一微秒，CUTLASS attention 约十七微秒，含准备算子约二十七点四微秒。TP4 的后续已测获益点分别是三十二和六十四。大、小 batch 来自两次任务，每一行是同卡比较，但整条曲线不能伪装成一次同机测量，也没有测所有中间 batch 来确定精确交点。

CUTLASS 还多了一层精度差异：它把 softmax 的未归一化概率舍入到 FP8，而原 Triton 用 BF16。对有效 FP8 Q 的数学参考，它仍有约百分之二点六的 NRMSE。我们用重复单页固定各 tile 的最大值，再用独立 FP8 概率参考，误差降到约百分之零点一六八；Q 等于零的精确均匀概率控制也接近这个量级。源码、SASS 和控制共同支持：主要额外误差来自概率侧舍入，不能把更快直接等同为同精度替换。

### 7:45–9:15｜怎样验收，候选结论到哪里为止（讨论点 6）

验收协议在候选实现前写出，内部进行了独立审查；真实外组挑战还没有举行。本次已确认按答辩材料和内部独立审查记录交付，这种替代不等于声称与别组交流已经发生。

gold 独立聚齐合法 KV，用 FP64 计算 QK、自然指数 softmax 和 PV，不调用 baseline，不使用同一个 split merge。先用 baseline 的十一、二十九号 seed 校准，在固定 floor、裕量与硬上限下冻结阈值，再用一百零一、三百零七号 seed 验证。阈值文件有协议和源码哈希，不能看完候选再放宽。

测试不仅有正常长序列，也包括短序列、跨页 dql、混合空 padding、非恒定且有 stride 的物理 token/head scale、强 logits、消去和明确的 future-token 泄漏探针。等价变换还会重排 top-k、一致置换物理页，并污染不可见位置。

当前 full heldout 中 baseline 一百六十八条、候选一百六十八条均通过冻结标准。性能另用预分配 graph，在相同 PDL 下随机交错 baseline 与候选，避免 Python 提交和固定测量顺序掩盖结果。六十四条性能记录还在 replay 后分别检查两端输出，一百二十八份输出均有限且通过独立参考检查。补充 graph 后输出检查的最终复测中，PDL 关闭的十六个形状都获益；PDL 开启时，TP1 batch 一、TP4 batch 四出现退化。因此这个结果支持条件改良，不支持无条件替换生产实现，更不等于服务尾延迟已经改善。

### 9:15–10:00｜复现与可审查性

复现材料包括 untouched 上游快照、基线 runner、候选、冻结验收、Modal 环境、原始计时样本、命令、NCU 报告与 SASS。基线、候选和 CUTLASS 的各类时间口径分开保存；失败的探索也保留，不只展示赢家。

这次最重要的不是挑出一个漂亮加速比，而是形成一条可查的证据链：先测清小 batch 的实际工作和瓶颈，再改一个可解释的成本，按冻结精度门槛验收，最后把形状、PDL、量化和准备成本都带回性能结论。尚未测过的共享 prefix、完整生产调度和真实跨组意见，不在当前结论范围内。

## 5 分钟可能问答

实际提问不固定。以下前五题按每题约 45–60 秒准备；后面的题目用于替换或追问，不要求五分钟内全部朗读。

### 1．你们为什么没有按题目预设，回答“交叉点就是十六”？

因为十六是上游注释和静态 dispatch 策略，不是待证明的公理。我们的同卡数据里，TP1 attention-only 在八就获益，加入本实验准备算子后是十六；TP4 在十六仍慢。改变计时边界会改变答案。我们保留上游策略和本轮实测之间的差异，而不回头改输入或忽略准备成本。B<16 的 CUTLASS 是强制底层实验，不代表生产 guard 已开放。

### 2．是不是 gold 写错，才让 CUTLASS 有百分之二点六误差？

gold 与 baseline 独立，而且我们分开比较原 BF16 Q 与量化后的有效 FP8 Q。当前 power-of-two scale 可精确反量化到 BF16，因此有效输入口径没有隐含差异。更关键的是，重复页控制中，加入独立的 P-FP8 舍入后误差从 0.026060 降到 0.001681；Q=0 时是 0.001663。再结合源码里的概率转换及对应 SASS，这支持主要差异是 P 舍入，而非简单 layout/scale 接错。它仍不是所有边界的形式证明，也没有因此放宽候选验收。

### 3．既然有 bank conflict，为什么不把 partial 和 merge 完全融合？

这两个问题不等价。改 merge 布局可以保留 split 并行；完全融合到一个 CTA 会减少 CTA 数，尤其 TP4 B1 从十六降到一。cluster 可保留部分并行，但有共驻留、DSM 生命周期和同步成本，并且 portable 八 CTA 与 S16 不能直接对应。本轮先解决有明确数据的 merge 成本，cluster 是未实现路线，不能拿它的理想收益和已完成代码比较。

### 4．你们说 TMA 不能表达间接寻址，但又跑通了 TMA，是否矛盾？

不矛盾。TMA 不会自己取 top-k 再查 block table；这两次依赖 load 是线程完成的。拿到物理页号后，页内数据仍是规则四维区域，可以用动态页坐标发起 TMA。实际 SASS 中普通 LDG 与 UTMALDG 同时存在。微实验只证明这条搬运路径正确，不能推断有 softmax、scale 和流水线后的完整 attention 必然更快。

### 5．是不是挑选 seed、关闭 PDL 才得到加速？

探索用 seed0，最终成对性能用 seed101/307，候选配置在最终测量前固定。每行两端使用同一输入、同一 PDL、预分配 workspace、同一 GPU，且随机交错 replay。PDL 开启有退化的形状照样报告，所以不主张全域胜出。正确性另用冻结 full suite，不能把性能样例里的粗略 sanity check 当完整验收。

### 备用 6．为什么不直接用理论 FLOP/HBM 时间判断收益上限？

Roofline 的吞吐是上界，对应时间是下界。小 batch 还存在 grid 不足、依赖访存、转换和 launch。用四 MiB 除标称带宽只能得到乐观的读取时间，不证明实际还剩多少可消除延迟，也不能独自证明“不值得做”。缓存温热时，逻辑 KV 字节甚至不全来自 HBM。

### 备用 7．标量 scale 都能移到外面，为什么不统一这样做？

精确算术里可以移动，不代表移动后保留同样的 BF16 舍入。非恒定 token/head K scale 必须对应每个 score；V scale 必须参与每个 token 的加权值。我们先明确 represented-input 契约，再决定是否接受变化，不能只根据代数等价删除中间舍入。

### 备用 8．NCU DRAM write 为零，是不是 workspace 成本不存在？

不是。store 指令和输出地址都存在；数据可能还留在 L2，没有在该计数窗口写回 HBM。理论 workspace 读写量、实际 L2 流量和 DRAM counter 是三种不同量。不能拿一次 DRAM write=0 删除理论数据依赖。

### 备用 9．PASS 是否证明可以投产？

只证明冻结输入域、参考和门槛下的回归检查通过。当前不包含模型质量、服务尾延迟、所有共享 prefix、完整生产调度或 sanitizer 的普遍保证。尤其 PDL 性能仍有退化，先保留条件适用范围；实际课堂活动没有举行，本次采用用户已同意的材料与内部审查替代交付范围。

## 讲稿用到的证据索引

|主题|可现场打开的证据|不能扩大成的结论|
|先 profile 后设计|[基线报告](experiments/BASELINE_PROFILE.md)、[完整原始矩阵](experiments/results/bench-b300-20260913T015939Z/measurements.json)、[代表 NCU](experiments/results/profile-matrix-b300-20260913T020433Z)|所有形状均为 HBM 带宽饱和|
|AI / split / cluster|[理论分析](analysis/C2_THEORETICAL_ANALYSIS.md)、[算术核对脚本](analysis/theory_checks.py)|逻辑流量等于实测 DRAM 流量；cluster 已验证|
|TMA 两级访问|[TMA 报告](experiments/TMA_EXPERIMENT.md)、[CUDA 源码](experiments/paged_copy.cu)、[原始结果](results/paged-copy-20260913T015626Z/result.json)|完整 attention 已因 TMA 获益|
|候选机制|[候选代码](candidate.py)、[候选 NCU CSV](results/candidate-profile-20260913T022426Z/selected.csv)|bank conflict 消除即全域性能改善|
|冻结 full 验收|[协议](validation/ACCEPTANCE.md)、[校准 manifest](results/candidate-verify-20260913T022150Z/calibration.json)、[168+168 PASS](results/candidate-verify-20260913T022150Z/candidate-heldout.json)|真实跨组挑战、生产全部输入或模型质量已通过|
|候选成对性能|[最终 64 条 paired 记录](results/candidate-paired-20260913T022747Z/paired.json)、[测量程序](experiments/candidate_workload.py)|热 graph 采样等于服务端到端效果；PDL 无条件更快|
|CUTLASS crossover 与 P 精度|[完整实验报告](experiments/CUTLASS_EXPERIMENTS.md)、[小 batch 对照](experiments/results/cutlass-b300-20260913T022105Z/measurements.json)、[P-FP8 控制](experiments/results/cutlass-b300-20260913T022105Z/probability-controls.json)|统一精确 cross16；更低精度路径通过候选冻结门槛|

## 六个讨论点的最短答法

1. **Tensor Core 有用，但 AI 与利用率不是一回事。** GQA16 带来 KV 复用，实际 HMMA 已存在；小 grid 和中间量降低达成率。
2. **不默认融合。** 合并代数允许树形/cluster；并行度与同步成本决定收益。本轮保留 partial，只改 merge。
3. **线程查页，TMA 搬页。** 页表追踪不能由 tensor map 自行完成；正确性微实验和 SASS 已验证这条组合路径。
4. **先定义 scale 与舍入口径。** 标量和 token/head scale 不同；地址用物理 token；原 Triton 是 FP8 存储加 BF16 dot。
5. **先有全矩阵与 NCU，再实施候选。** 最小 batch underfill，较大形状有明显读取压力；FP8 转换与 merge bank conflict 都有实际证据。
6. **独立 FP64、先校准冻结、后 heldout。** 当前 full 168+168 PASS；实际课堂活动未举行，交付采用用户同意的材料与内部审查替代范围，未覆盖生产域不外推。
