# C2 MiniMax M3 MSA decode 理论分析

分析日期：2026-09-13。分析对象是题目指定的 vLLM `d4da0c5` 快照及本地 harness；本文不把这个固定版本的行为描述成所有后续上游版本的行为。

**核心判断：小 batch MSA decode 的问题不能简化为“Tensor Core 没用”或“两个 kernel 应该融合”。它在 GQA 复用、split 并行度、局部 softmax 状态、两级索引、FP8 反量化和启动开销之间做取舍。batch=16 是上游给出的经验 dispatch 门槛，当前材料不足以将其解释为硬件或算法定律。**

本文逐项分析任务的六个讨论点，给出数学推导、源码事实、成本模型、条件假设与验收草案。按照 [TASK.md 的要求](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/TASK.md:23)，优化设计须在 profile Triton 基线后决定。因此这里列出的融合、cluster、TMA 等路线是等待测量检验的候选，不是已选定的实现方案。

本次没有新增 GPU profile、CUDA 性能数据或 CUTLASS 对拍。在 c2 目录及相关实验输出中未发现已有 C2 测量结果。附带脚本只做 CPU 数学检查与逻辑流量计算，不能充当任务第一层的测量交付。

**1．先固定算子语义：稀疏的是 KV 页集合，GQA 复用仍然存在。**

定义：

- $B$：独立请求数；$d_q$：每请求 decode query 数，普通 decode 为 1，投机验证可大于 1。
- $R=Bd_q$：总 query token 数；$H_q,H_{kv}$：每卡 query/KV head 数；$G=H_q/H_{kv}$。
- $D=128$：head dimension；$P=128$：每页 token 数；$K_b=16$：最多选择的页数。
- 对长度充分、所选页均完整的 query，每个 KV head 选择 $L_{sel}=K_bP=2048$ 个 token。

题面典型 TP1 形状为 $H_q=64,H_{kv}=4,G=16$；TP4 为 $H_q=16,H_{kv}=1,G=16$。TP 切分了每卡工作量，但这两个场景的组内复用比例 G 相同。

物理 cache 的布局是：

$$
KV[\mathrm{physical\ page},\mathrm{KV\ head},\mathrm{position},2D],
$$

最后一维前 D 个元素是 K，后 D 个元素是 V。同一 KV head 对应的 G 个 query heads 共享这组 K/V 与所选逻辑块，但注意力权重各自计算。

对 flatten 后的 query token $t$：

$$
r=\lfloor t/d_q\rfloor,\quad u=t\bmod d_q,
$$
$$
qpos=seq\_lens[r]-d_q+u,\quad kv\_len=\max(qpos+1,0),
$$
$$
k_{real}=\min\left(K_b,\left\lceil kv\_len/P\right\rceil\right).
$$

只读取 `topk_idx[kv_head,t,:k_real]`。这是由可见长度决定的有效前缀，**不是用 -1 sentinel 终止**。对于其中的逻辑块号 $b$：

$$
page=block\_table[r,b],\quad pos=bP+n,\quad 0\le n<P,
$$

仅 $pos<kv\_len$ 的 token 参与当前 query 的 attention。物理 page 号不表示 token 的时间先后，因果判断必须用逻辑位置。

令 $\mathcal I_{t,h}$ 为选中且满足因果条件的 token 集合，算子是：

$$
z_{a,j}=\frac{q_a^Tk_j}{\sqrt D},\qquad
p_{a,j}=\frac{e^{z_{a,j}}}{\sum_{l\in\mathcal I_{t,h}}e^{z_{a,l}}},\qquad
o_a=\sum_{j\in\mathcal I_{t,h}}p_{a,j}v_j.
$$

这里 $a$ 是组内 query head，不是 query 的时间位置。main attention 的数学目标是在给定稀疏集合上计算标准 softmax attention；“稀疏”是对全上下文注意力的选择，不能再把结果和不做稀疏选择的 dense 全上下文 attention 当成同一数学算子对拍。

源码依据见 [cache 布局](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/vllm_msa_ref/sparse_attn.py:10)、[decode query/长度映射](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/vllm_msa_ref/sparse_attn.py:280) 与 [间接访问/因果 mask](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/vllm_msa_ref/sparse_attn.py:325)。

活跃 top-k 前缀应包含互不重复的合法逻辑块。重复块不会被 main attention 自动去重：某 token 重复 m 次，会使它的有效 softmax 质量增大 m 倍，等价于该 token 的 logit 增加 $\ln m$，因此重复不是无害冗余。harness 保证选入当前块；indexer 则要在当前块属于强制 local 集合且强制集合能被 top-k 容纳的配置下讨论这个保证，不能推广到任意 init/local 配置。main attention 不负责补上当前块。

Indexer 为每个可见块计算 index-Q/index-K 点积的 max，再结合 init/local 强制块规则选 top-k。不要根据文件头的旧“single shared index head”注释推断所有 KV heads 共享一份 top-k：decode wrapper 实际断言 `num_idx_heads == num_kv_heads`。index-K 存储与 index-query head 数是不同概念，见 [index_decode_score](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/vllm_msa_ref/index_topk.py:759)。

整个推理步骤还包括 indexer、top-k、query 投影/量化、KV 写入等。harness 已给定合成 top-k，仅测 main attention。虽然 main attention 每 query 最多读 2048 个 token，indexer 仍可能扫描更长上下文，不能据此宣称整个 decode 对上下文长度完全不敏感。

**2．讨论点一：arithmetic intensity 必须计入 GQA 复用。**

每个 query、每个 KV head，在选中 L 个 token 上计算两次矩阵乘：

$$
QK^T:[G,D]\,[D,L]\to[G,L],
$$
$$
PV:[G,L]\,[L,D]\to[G,D].
$$

每次 FMA 按 2 FLOP，主矩阵乘总量为：

$$
F=4RH_qLD.
$$

设一个 KV 元素占 $s_{kv}$ bytes。若同组 G 个 query heads 共享一次 KV 读取，KV payload 是：

$$
B_{KV}=2RH_{kv}LDs_{kv}.
$$

于是只计 KV 的理想强度为：

$$
AI_{KV}=\frac{F}{B_{KV}}=\frac{2G}{s_{kv}}.
$$

| KV 格式 | 每元素字节 | G=16 时理想 KV-only AI |
|---|---:|---:|
| BF16 | 2 | 16 FLOP/B |
| FP8 | 1 | 32 FLOP/B |

若把每个 query head 独立实现为逐行 dot，并让每个 head 都重新读取同一组 KV，上述复用可能丢失，理想强度就会退化为 BF16 约 1、FP8 约 2 FLOP/B。即使重复读取命中 L2，也会增加 L2/L1/发射侧压力；不能把“HBM 可能命中缓存”解释为重复工作免费。

对 TP1、B1、d_q=1、L2048：主矩阵乘为 67,108,864 FLOP，BF16 KV payload 为 4 MiB，FP8 为 2 MiB；TP4 均为其四分之一。Q 与 output 均 BF16 时各为 TP1 16 KiB、TP4 4 KiB。加入一次 Q 读/输出写、暂不含 split workspace，AI 分别约为 15.876 与 31.508。

这解释了 Tensor Core 为何仍有用武之地：decode 时间维虽然只有一个 query token，组内仍有 16 个 query heads 可以形成 M=16 的矩阵，且复用同一 KV。基线已使用 `tl.dot`，不是一份纯标量 CUDA Core 算法；具体 lowering/指令仍应检查所用 Triton 版本生成的 PTX/SASS。对于G小于16的其他支持形状，源码还会把head tile至少补到16，不能继续把有效FLOP直接当成实际发出的矩阵工作量；上游scale测试的G2就不是典型G16性能形状。

它也解释了为什么 batch 增大不等于 arithmetic intensity 必然增大：一般不同请求有不同 KV 数据，B 主要增加独立工作，不像同一权重 GEMM 那样天然按 B 复用同一矩阵。共享前缀的物理页复用是另一种有条件的缓存机会，需要真实 workload 证据。

这与 [handout 4.5 的瘦 GEMM](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/handout/src/assignment02.md:509) 有联系，但不可混为一谈。投影 GEMM 的 M 是共享权重上的 token 数；MSA 单页乘法的 M 可以是 GQA 组内 head 数，跨请求 KV 通常各不相同。

**3．单页矩阵形状与 SM100：不能仅看到 M16 就判定不适合。**

以每页128 token、G16、D128 为例：

| 运算 | 原始逻辑 $(M,N,K)$ | 转置后的逻辑 $(M,N,K)$ |
|---|---|---|
| $QK^T$ | (16,128,128) | $KQ^T$：(128,16,128) |
| $PV$ | (16,128,128) | $V^TP^T$：(128,16,128) |

普通 BF16 `tcgen05.mma.cta_group::1` 的 M128、N16 合法，逻辑 K128 可以由八个 K16 步组成。因此在数学形状上，可以避免直接把 M16 填充成 M64 的浪费。形状依据见 [NVIDIA PTX tcgen05 matrix shape](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-matrix-shape)。

但转置把 softmax 的 token 归约方向也转了：原来沿 logits 的列求 max/sum，现在要沿转置 logits 的行维归约。TMEM accumulator 如何被 SIMT 读取、归约、求 exp、重新转换并提供给下一次 PV，是主要的数据流问题。完整链路是 `QK → mask/softmax → probability cast → PV`，不能拿两个孤立 GEMM 的速度相加当作 attention 的速度。

GQA=16 已给基线一个自然的矩阵 tile。把四个不同请求简单拼成 M64，若它们使用不同 K/V，会计算错误的跨请求交叉项或需要额外块对角 padding；只有确有共享操作数时才能直接拼接获得有效复用。

投机验证的 $d_q>1$ 与增加独立请求B也不同：同一请求的多个query访问同一物理KV池，可能有更多复用。若selected集合相同，可以考虑把query/head合成更高的M；但top-k按每个query给定，集合和因果上界可能不同，使用并集时要保留各query自己的稀疏与因果mask。这里存在条件性的tile机会，不能默认所有投机query都共享16个相同块。

FP8 又有独立的指令约束。当前 Triton 对 FP8 KV 先转换到 `q.dtype`，乘 scale 后再转换一次，BF16 query 下的 dot 输入是 BF16。不能把它称作原生 FP8 Tensor Core 基线。CUTLASS 接口需要 `query_fp8`，但底层实现不在本快照，实际指令、packing 和路径仍要核查；单凭接口不能承诺具体吞吐。

因此讨论 Tensor Core 应分成三个问题：是否能保留 GQA 数据复用；矩阵形状是否有有效映射；非矩阵部分和片上往返是否抵消收益。前两个可以先做理论分析，第三个需要测量。

**4．split-K 为什么数学上成立：合并的是 softmax 质量。**

对一个 query head，把所选 token 集合分成互不相交的 split $\mathcal I_c$。定义：

$$
Z_c=\sum_{j\in\mathcal I_c}e^{z_j},\qquad
o_c=\frac{\sum_{j\in\mathcal I_c}e^{z_j}v_j}{Z_c},\qquad
\ell_c=\log_2Z_c.
$$

则全局输出为：

$$
o=\frac{\sum_cZ_co_c}{\sum_cZ_c}
=\sum_cw_co_c,
$$
$$
w_c=\frac{2^{\ell_c-\ell_{max}}}{\sum_d2^{\ell_d-\ell_{max}}}.
$$

所以基线第一步存储的是**局部归一化输出和 log2-LSE**，第二步按 LSE 重新分配各 split 的质量。不是把局部输出直接求和、也不是平均。源码见 [merge kernel](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/vllm_msa_ref/sparse_attn.py:419)。

一个简单反例：三个 token 的 logits 是 [0,0,10]、values 是 [0,0,1]；将前两个分到一个 split，最后一个分到另一个 split。两个局部输出分别为 0 与 1，直接平均得到 0.5，而正确输出为：

$$
\frac{e^{10}}{2+e^{10}}\approx0.9999092084.
$$

即使 split 长度相等，仍不能省略 LSE 权重：例如仅两个token，logits为[0,10]、values为[0,1]，每个split各一个token，正确输出是 $\sigma(10)\approx0.9999546$，直接平均仍为0.5。

基线在每个 split 内仍按页进行 online softmax。令 $x_j=z_j\log_2e$，维护最大值 m、以当前最大值缩放的 numerator 向量 a，以及 log2-LSE $\ell$。加入一页后：

$$
m'=\max(m,\max_{j\in new}x_j),
$$
$$
a'=2^{m-m'}a+\sum_{j\in new}2^{x_j-m'}v_j,
$$
$$
\ell'=m'+\log_2\left(2^{\ell-m'}+\sum_{j\in new}2^{x_j-m'}\right).
$$

最终 $o_c=a\,2^{m-\ell}$。减最大值用于控制指数范围，exp2/log2 的底数必须与 LSE merge 一致。具体实现见 [decode softmax/PV](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/vllm_msa_ref/sparse_attn.py:355)。

从归约代数看，也可维护三元组 $(m,\lambda,a)$，其中 $\lambda=\sum_j2^{x_j-m}$。两个三元组合并时共同换到 $m'=\max(m_1,m_2)$，分别用 $2^{m_i-m'}$ 重标定 $\lambda_i$ 和 $a_i$ 后相加。这个结合律支持树形 merge、cluster 或多级归约；但有限精度下，改变树形与 split 数会改变舍入位置，不保证逐位相同。

当前源码的概率在 PV 前转为 V dtype，partial output 存 q.dtype，LSE 存 fp32。对于 BF16 Q，这意味着局部归一化输出先量化为 BF16，之后 merge 再加权。若把融合版本改为保存 fp32 numerator，通常是另一条更高精度的数值路径，而非与现有结果逐位等价。

**5．空 split 与全空请求必须分开定义。**

当序列短、实际有效页不足16时，shape 决定的 split 数可能大于有效页数。一部分 split 没有任何 token，数学上可用 $Z_c=0,\ell_c=-\infty,o_c=0$ 作为空贡献。若该 query 还有其他非空 split，权重为0，合并正常。

源码专门处理了活跃请求里的空 split，避免输出 NaN 后出现 `0 * NaN`。但是当整行请求为空，例如 CUDA Graph padding 的 `seq_len=0`，所有 LSE 都是 $-\infty$，merge 的 $\ell_c-\ell_{max}$ 会成为 $-\infty-(-\infty)$，产生 NaN。这一点源码注释也明确承认，见 [空 split 处理](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/vllm_msa_ref/sparse_attn.py:391)。

这不是“真实活跃请求输出允许 NaN”。对真实请求应要求非空可见集合和有限输出；对纯 padding 行应在验收契约中明确：输出被忽略，还是新实现定义写0。测试必须区分 active mask，不能对被声明为忽略的 padding 行盲目套用整个张量的有限性要求，也不能让 padding 的写入污染邻接活跃行。

现有 `ref_sdpa.py` 不覆盖这个角落：它对空集合会执行空列表 `torch.cat`，且没有与基线相同的 `max(kv_len,0)` 处理。它适合作为普通输入的起点，不能直接称为覆盖全部 CUDA Graph padding 的 gold。

**6．实际 split 数与 CTA 数：这是 crossover 分析的起点。**

基线使用 shape-constant 的 split 数，定义目标 grid 为256：

$$
t=\max\left(1,\min\left(K_b,\left\lfloor\frac{256}{RH_{kv}}\right\rfloor\right)\right),
\qquad S=2^{\lfloor\log_2t\rfloor}.
$$

partial grid 为 $RH_{kv}S$ 个 CTA；merge grid 则是 $RH_q$ 个 CTA。每个 partial CTA 处理一个 query token、一个 KV head、若干连续 top-k 槽，组内 G 个 query heads 一起计算。S 的选择取决于 shape，不依赖运行时 top-k 的数值，便于 CUDA Graph 固定 grid。依据 [wrapper 的 split 选择](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/vllm_msa_ref/sparse_attn.py:647)。

普通 decode 的精确表如下：

| 每卡形状 | B | S | 每 split 页数上限 | partial CTA | merge CTA |
|---|---:|---:|---:|---:|---:|
| TP1：64Q/4KV | 1 | 16 | 1 | 64 | 64 |
| TP1 | 4 | 16 | 1 | 256 | 256 |
| TP1 | 8 | 8 | 2 | 256 | 512 |
| TP1 | 16 | 4 | 4 | 256 | 1024 |
| TP4：16Q/1KV | 1 | 16 | 1 | 16 | 16 |
| TP4 | 4 | 16 | 1 | 64 | 64 |
| TP4 | 8 | 16 | 1 | 128 | 128 |
| TP4 | 16 | 16 | 1 | 256 | 256 |

结论有四个：

第一，TP1 在 B4 就有256个 partial CTA；TP4 到 B16 才有256个。因此不能把“TP1/TP4 的 CUTLASS crossover 都在16”解释为两者都到16才拥有同样数量的 CTA。

第二，CTA 数量相同不意味着负载相同：TP1 B4/8/16 的每个 partial CTA 分别处理1/2/4页，Q 复用、页间 online softmax、启动成本占比都发生变化。merge grid 也不是固定256。

第三，以 C1 已保存的 B300 148 SM probe 作硬件情景参考，TP4 B1 的16个 partial CTA 最多覆盖约10.8%的 SM，TP1 B1 的64个最多约43.2%。这是静态覆盖上限，不能冒充 C2 的 ncu occupancy 实测。该硬件记录见 [B300 probe](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c1_flashkda/results/environment-20260910T041816895557Z.json:29)。

第四，短序列与 Graph padding 可能使很多已发射 CTA 做空工作。需要同时记录请求数、有效请求数、graph bucket、d_q 和 real_topk；只报 batch 容易把空行、投机 query 与实际独立工作混在一起。

这和 C1 的状态链有所不同：C2 的不同 KV split 可以真正独立计算，然后通过结合性归约合并。减少 split 会降低归约开销，但也主动减少并行任务；不存在“越少 kernel、越少 split 就必然越快”的单向结论。

**7．把 partial、merge 和重复 Q 读取计入流量。**

以下假设 Q/输出/partial 为 BF16、LSE 为 fp32、所有选中页完整，暂不含索引、scale、cacheline overfetch、spill 与跨 query 复用。定义单次完整 Q payload：

$$
Q_b=2RH_qD.
$$

partial workspace 大小为：

$$
W=S RH_q(2D+4).
$$

S 个 partial 会逻辑上各读一次 Q；workspace 写一次、merge 读一次；最终 output 再写一次。所以当前算法的逻辑请求模型为：

$$
B_{req}=B_{KV}+SQ_b+2W+Q_b.
$$

这是显式数据 payload 的模型，不是 DRAM 实际事务计数：Q 重读可能命中 cache，partial 常可能命中 L2，物理页分布和事务粒度也会改变实际流量。

TP1 的计算结果如下，表中的 FP8 列尚未计入逐 token scale：

| B | S | workspace 大小 | BF16 请求量 MiB | FP8 请求量 MiB | BF16 AI | FP8 AI |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 260 KiB | 4.7734 | 2.7734 | 13.4075 | 23.0761 |
| 4 | 16 | 1040 KiB | 19.0938 | 11.0938 | 13.4075 | 23.0761 |
| 8 | 8 | 1040 KiB | 35.1563 | 19.1563 | 14.5636 | 26.7276 |
| 16 | 4 | 1040 KiB | 67.2813 | 35.2813 | 15.2197 | 29.0239 |

TP4、B1 的 workspace 为65 KiB，请求量为 BF16 1.19336 MiB、FP8 0.69336 MiB；B1/4/8/16 的 S 都是16，因此各自 AI 与 TP1 的 S16 列相同，流量随 B 增长。

FP8 每物理 token/head 的 K/V scale 若都存 fp32，会额外读取：

$$
B_{scale}=8RH_{kv}L.
$$

相对于 FP8 K+V payload，其比例为 $8/(2D)=3.125\%$。TP1 B1 为64 KiB，计入后请求 AI 从23.076降至22.567。字节比例不算大，但访存依赖、地址计算和反量化指令的成本还应单独观察。

两级索引本身的 payload 更小：每个选中块一个 int32 top-k 与一个 int32 page-table entry，约 $8RH_{kv}K_b$ B；TP1 B1 为512 B。**小字节量不等于小延迟**，因为它们位于 K/V 加载之前的串行地址依赖链上。

若仅删除 partial 的一写一读，TP1 B1 理论上省去520 KiB，占 BF16 请求量约10.64%、FP8约18.31%。这不是融合速度上限：融合还可能省 Q 重读和 launch，但可能丢并行度、引入共享内存/DSM 往返；而原 partial 可能已命中 L2。它只是判断“只靠少一次全局中间结果往返”能省多少 payload 的账目。

还有一个特殊点：当 S=1 时，数学上不需要再跨 split merge，但当前 wrapper 仍发 merge kernel并分配 workspace。TP1普通decode从B33起进入S1，现有harness的离散扫描在B64首次覆盖这个区间；TP4则从B129起进入S1。这不是题目B<16的主要现象，却说明shape-specific简化要先识别生效域。

**8．讨论点二：融合的收益和代价，必须放在同一张账上。**

对于两 kernel 路径，可写成示意模型：

$$
t_{base}=t_{partial}+t_{merge}+t_{exposed\ launch/gap}+t_{interface}.
$$

PDL、CUDA Graph 和 host/device 重叠会影响暴露出来的 launch/gap，不能把所有孤立开销直接相加。对融合路线，应检查：

$$
\text{节约的 merge/往返/启动暴露成本}
>
\text{新增同步/片上搬运/资源占用/并行度损失}.
$$

可以讨论但尚未由 profile 选定的路线如下：

| 候选 | 理论收益来源 | 需要挑战的反例 |
|---|---|---|
| 单 CTA 处理一个 query/KV head 的全部16页 | 一次 Q 读取、无跨 CTA merge、无需partial全局往返 | TP1 B1只剩4 CTA，TP4 B1只剩1 CTA；大部分SM可能空闲 |
| 保留多个 split、优化独立 merge | 保留并行度，可能降低merge发射/读写成本 | 若merge时间很小，端到端收益有限；不能根据小FLOP自动认定它便宜 |
| cluster 内多个 CTA 做 split，再经 DSM 归约 | 保留部分split并行度并减少全局中间量 | 共调度约束、DSM延迟、barrier和资源分配可能抵消收益 |
| 一个 CTA 内多 warp/subtile 处理多个页 | 消除CTA间归约，仍可在CTA内并行 | 独立SM任务数降低，多个logit/accumulator增加寄存器与SMEM压力 |
| 两级 merge 或批量合并多个 head | 改善很多小merge CTA的调度/内存效率 | 更大的工作tile可能增加寄存器与不规则访问；低B时有不同最优点 |

**现有生产路径已条件使用 PDL。** wrapper 检查平台是否支持，partial 可以触发 dependent launch，merge 在读 partial 前执行 `gdc_wait`。提前发射是调度机会，不意味着 partial 数据已经写完；trigger 与数据可消费是两个时点。源码见 [PDL 分支](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/vllm_msa_ref/sparse_attn.py:641)，语义见 [NVIDIA PDL 文档](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html)。

因此不能以“现有两个 kernel 一定有完整串行 launch 空隙”作为融合收益依据。harness 的 shim 强制关闭 PDL，所以它与生产 baseline 在性能机制上已有区别；语义相同不意味着延迟相同。

**9．cluster 归约与 mbarrier 同步的协作范围。**

softmax 部分状态的结合性使 cluster 归约在数学上可行，mbarrier负责同步而不直接计算归约。硬件正确性要求：生产者先建立有效的共享存储；消费者知道所有必要部分已经完成；不同阶段使用正确的 barrier phase/计数；所有远端读取结束前，拥有该共享内存的 CTA 不得提前退出。

关键边界包括：

- portable thread-block cluster 上限为8个 CTA；本基线小 batch 的 S16 不能未经核实直接映射成一个16 CTA portable cluster。可以研究一个CTA处理多页或不同层级，但那会改变并行与存储账目。[Thread Block Clusters](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html#thread-block-clusters)
- cluster 在一个 GPC 范围共调度，共享资源与 DSM 的访问范围由该协作组确定；它不是任意全GPU CTA之间的低成本共享内存。
- `mbarrier` 本身不让普通 grid 中任意 CTA 保证共驻留。远端 shared barrier 支持 arrive，不能把它当作可任意 remote wait 的对象；可以由 producer 对 consumer 本地 barrier arrive，由 consumer 在本地等待。[高级同步原语](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#advanced-synchronization-primitives)
- 正式的 `grid.sync()` 需要 cooperative launch，并满足整个 grid 的资源限制。普通 kernel 中让所有 CTA 自旋等待尚未调度的 CTA，可能形成死锁，不能代替合法的全局同步。[Cooperative Groups](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html#large-scale-groups)

普通的软件arrive也不自动意味着先前的TMA/tcgen05已经完成，barrier必须实际追踪相应异步操作；generic shared写入再供异步引擎读取时，还要满足对应proxy可见性规则。完成通知、线程间同步和跨proxy可见性不能互相替代。

即使同步正确，cluster 也不自动比 L2 中的 partial 往返快。B1 下 cluster 很少、每组受调度约束；B16 下可能已有很多独立 CTA，强行绑成协作组会减少调度自由度。需要测 cluster occupancy、活跃CTA、等待、DSM/SMEM请求，以及原方案 merge 与 gap 的实际时间份额。

讨论融合时还要固定数值契约：若采用 fp32 numerator 而非 BF16 局部归一化输出，精度变化要单列；若合并时只保留局部最大值而没有 denominator/等价LSE，则在数学上信息不足。

**10．讨论点三：两级间接寻址并不排斥逐页 TMA。**

地址链分成两部分：

$$
\mathrm{slot}\to b=topk[h,t,slot]\to page=block\_table[r,b],
$$
$$
addr=KV_{base}+page\cdot stride_{page}+h\cdot stride_h+n\cdot stride_n+d\cdot stride_d.
$$

前半部分是运行时内容相关的指针/索引追踪，后半部分是在已知物理页上的规则张量坐标。

**TMA tensor map 能描述物理 KV 池的规则布局；不能替 SM 执行 top-k→block-table 两级查表。** 线程得到 page 后，可以把它作为动态坐标发起该页规则 tile 的 TMA 搬运。因此“一个普通 tensor map 一次自动解析任意top-k页链”不成立，“paged KV 所以不能用 TMA”也不成立。依据 [CUDA Tensor Map API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)。

K/V 在最后维拼接到同一个物理池，最后维为2D，可以分别按 K 或 V 的坐标范围搬运，或研究符合指令/布局约束的合并搬运。这种布局描述不意味着需要启用TMA的interleave mode。descriptor 的 base、stride、box 与 swizzle 需要满足合法性要求；global layout 合法不等于已经得到适合 MMA 的 shared layout，不能跳过对齐与 swizzle 分块检查。

TMA 的 gather4 也不是任意页列表遍历引擎。它针对二维 tensor 的四条非连续行、共享列起点；im2col 则表达规则卷积坐标变换。即使使用这些模式，page-table 中的动态 page 号仍须先取得。参见 [PTX gather4/scatter4](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-tiled-scatter4-gather4-mode)。

可供 profile 后比较的搬运手段有：

| 手段 | 能表达什么 | 可能的代价 |
|---|---|---|
| 当前指针化/vectorized load | 完整任意已计算地址与逐元素mask | 地址/加载指令、scoreboard和寄存器压力 |
| `cp.async` | 线程算出地址后的global→shared异步拷贝 | 多条小拷贝、alignment、completion wait及消费者同步 |
| 每页动态坐标 TMA | 已知page上的规则大tile搬运 | scalar查表不能消失；TMA发射、barrier、SMEM staging |
| 预先压成连续 selected-KV buffer | 下游得到连续工作集 | 多一个gather阶段，额外写读，buffer空间；一次decode是否能摊销须测 |

异步拷贝正确性和机制见 [NVIDIA Asynchronous Data Copies](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html)。

单个 KV head 的整页 K+V payload 为 BF16 64 KiB、FP8 32 KiB；双 buffer 分别为128/64 KiB，还未计 Q、logits、输出累加器、反量化缓冲与barrier。若FP8先完整解量化到BF16，则片上容量不能仍只按FP8计算。这是容量预算，不是要求一次必须搬整页或一定使用双 buffer。

**TMA 的物理越界 zero-fill 不能替代逻辑 causal/tail mask。** 尾页/投机未来token的地址可能完全在已分配的物理池内，descriptor并不知道它对当前query不可见。若整页搬入，仍须按逻辑位置遮蔽 logits；而无效 V 也应按约定处理，避免概率0乘未初始化NaN污染PV。当前基线对masked K/V读取使用0，候选应保持等价行为或明确底层数据契约。

索引预取还必须遵守有效slot边界：`slot>=real_topk`的尾槽可能是无效块号，不能先访问越界block table，再指望KV tensor map的OOB机制保护。当前baseline由有效循环上界避免这类访问；提前解析下一页时也要保留这个语义。

测量上应将“地址链延迟”与“KV带宽”分开：可以比较相同物理页分布下预解析page与两级查表，比较连续/随机物理页，观察 L2 命中、long scoreboard、发射和实际DRAM字节。索引只有512 B并不能证明它不在关键路径。

**11．讨论点四：FP8 scale 放在哪里，先区分数学等价与舍入等价。**

用 $\widehat K_j,\widehat V_j$ 表示FP8存储的数值，实际反量化为：

$$
K_j=a_j\widehat K_j,\qquad V_j=b_j\widehat V_j.
$$

scale 是解量化时的乘数；上游测试先用原值除以scale再转FP8，之后用FP8数值乘scale恢复，见 [FP8测试的数据构造](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/vllm_msa_ref/test_sparse_attn_fp8_scale.py:49)。其他库可能把“scale”命名为相反方向的量，应以这个接口的实际乘除定义为准。

本地 Triton 支持两种有scale模式：K/V都是scalar；或二者均为 `[num_kv_heads,max_physical_tokens]` 且shape相同。K/V scale必须同时提供或同时不提供。逐token索引为：

$$
physical\_token=page\cdot128+n.
$$

不是 $b\cdot128+n$，不是query位置，也不是top-k槽号。scale必须跟随**物理页数据**做一致重映射。具体访问在 [K反量化](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/vllm_msa_ref/sparse_attn.py:342) 与 [V反量化](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/vllm_msa_ref/sparse_attn.py:371)，接口校验见 [_kv_scale_args](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/vllm_msa_ref/sparse_attn.py:476)。

精确算术下，scale可移动的位置如下：

| scale类型 | 可以移到哪里 | 不成立的移动 |
|---|---|---|
| 全部selected K共有scalar a | 合并到 `sm_scale`，或同组Q乘a | 把a移到softmax后/最终输出后 |
| 每token K scale $a_j$ | 乘每个logit列，必须在max、exp和LSE之前 | 用统一标量代替不同列scale；只缩放最终LSE |
| 全部selected V共有scalar b | 最后输出乘b | 将它混入softmax denominator |
| 每token V scale $b_j$ | 先乘V，或在PV前把概率因子改为 $p_jb_j$ | PV完成后用单一scale恢复；重新归一化 $p_jb_j$ |

K scale改变softmax分布；V scale改变加权输出值，二者不能交换。尤其 $p'_j=p_jb_j$ 只用于numerator，denominator仍由原始未乘V scale的概率计算。若把p'再归一化，计算的是另一个attention。

反例可用一个独立的一维玩具问题表示，取D=1、`sm_scale=1`：两token raw K都为1、query为1、K scale分别为1和2、V为[0,1]，正确输出为 $\sigma(1)=0.7310586$，忽略不同K scale则为0.5。若两token logits都为0、raw V为[0,1]、V scale为[1,3]，正确输出为1.5，最后统一乘第一个scale得到0.5。这些数值仅用于说明scale代数，未沿用D128的缩放。

当前Triton的精度路径是：FP8→q.dtype，乘scale，再转回q.dtype；之后dot和softmax做各自的累加/转换。将scale移到fp32 logit上或最终输出后，会改变此前BF16反量化舍入，因此仅能主张精确代数等价，不能主张bitwise等价。FP8的表示与缩放背景可读 [Transformer Engine FP8说明](https://docs.nvidia.com/deeplearning/transformer-engine/examples/fp8_primer.html)，本题具体规则以vendored源码为准。

**12．现有 FP8 测试究竟覆盖了什么？**

上游测试中的 `scalar` 与 `per_token_head` 两种模式，实际分别使用一个常数scalar或填满同样常数的二维表。K_SCALE=0.25、V_SCALE=0.5；仅有一个KV head、一个物理页，decode长度64/128，Q head数为2，topk=1。测试比较反量化后的BF16路径，并验证不应用scale会产生明显差异，容差为 `rtol=atol=2e-2`。见 [FP8 decode scale test](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/vllm_msa_ref/test_sparse_attn_fp8_scale.py:135)。

这能检出“完全漏掉scale”，但不能检出：

- 把物理token下标错误写成逻辑token下标：单页时二者可能相同，常数表更无法区分。
- 使用错误的KV head scale：只有一个head。
- 把逐token scale错误当scalar：表内数值全相同。
- 把scale移到不正确的计算层：某些常数/2的幂的样例会隐藏问题。
- 多split LSE合并、页顺序变化、d_q跨页、Graph padding等交互。

这里不是否定原测试用途；它是一个scale语义回归测试，而非完整C2验收集。本题需要在它基础上构造多页、多KV head、非恒定且不全为2的幂的scale，以及被打乱的物理页。

另一个公平比较问题是：CUTLASS wrapper接收 `query_fp8` 与scalar q/k/v scale；Triton通常接收BF16 Q与FP8 KV，并支持逐token/head scale。两者不是只替换一条矩阵指令，Q的量化误差、转换开销和支持域必须单列。源码见 [backend中的不同调用](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/vllm_msa_ref/sparse_attention_msa.py:236)。

**13．讨论点六：一份可提交讨论的验收草案。**

验收应至少区分三层，而不是一个全局相对误差数：

| 层次 | 参照 | 评价对象 |
|---|---|---|
| 索引/算子语义 | 直接按top-k、block table、causal规则聚齐selected K/V，显式fp64 softmax | 是否读取正确数据、GQA分组/尾块/投机位置是否正确 |
| kernel数值实现 | 固定同一份有效Q/K/V数值，与高精度attention比较 | partial舍入、LSE、exp近似、归约树和scale位置带来的误差 |
| 量化与应用质量 | 未量化输入参考、同checkpoint模型任务 | FP8编码、Q量化、scale策略对输出/模型质量的影响 |

对FP8第一种数值参考，可以先严格按当前基线规则反量化并舍入到BF16，再升到fp64计算attention；它隔离kernel计算误差。另一种参考直接把FP8数值与scale在fp64中相乘，它同时包含改变反量化舍入的效果。二者都可用，但报告必须命名清楚，不能在同一误差表中偷换。

现有 `ref_sdpa.py` 实际使用 `.float()` 和显式matmul/softmax，最后再转Q dtype，并不是独立的fp64 gold。使用GPU float32 matmul还应明确TF32等精度设置；仅函数名带“reference”不能证明精度充分。实现见 [ref_sdpa.py](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/harness/ref_sdpa.py:13)。

建议先讨论以下输入契约：D128、page128、topk16；TP1/TP4及明确支持的其他GQA形状；Q dtype、KV dtype与scale模式；活跃前缀必须有效无重复；seq_len包含当前query；空请求仅作为显式padding；output与stride支持范围写清楚。数值域也应明确，例如scale采用有限正fp32值，Q/K/V及反量化中间值在约定dtype内可表示，另将超范围/NaN输入作为单独鲁棒性测试。上游shape/device检查并未定义全部数值域，不能把人为构造的数值溢出与索引/同步错误混为一谈。若候选只支持这个子域，应明确fallback，而不是默默处理未验证输入。

测试矩阵建议覆盖：

| 维度 | 需要覆盖的值或结构 |
|---|---|
| batch | 1/4/8/16，加15/17观察dispatch边界；active batch与graph bucket分开 |
| 每卡heads | 64Q/4KV、16Q/1KV；若宣称通用，再加G非16和多KV head |
| 序列长度 | 1、127/128/129、255/256/257、2047/2048/2049，以及8k/更长上下文 |
| decode query长度 | 1、2、4及声称支持的上限；必须含query位置跨页 |
| 稀疏索引 | 顺序/乱序、远距物理页、强制当前块、有效prefix长度小于16 |
| 数据 | Q=0、常量V、随机正负、输出消去、logit几乎并列、极尖softmax、不同幅度 |
| FP8 scale | scalar、真实变化的physical-token/head表；多页多head、非2幂scale、明显不同K/V scale |
| 空与尾部 | 活跃行空split、全空padding、尾页未来token、未selected页污染测试 |
| 布局 | baseline接受的strided topk view，page table边界，实际接口的contiguous约束 |

结构性检查比普通随机样例更容易揭示错误：

1. 在同一输入内打乱selected块顺序，数学结果不变，允许浮点容差内差异。
2. 一致置换物理KV页、block table和物理scale，结果不变。
3. 改变未selected页或未来token，活跃输出不应改变；若在合法内存范围放置NaN作为污染值，应明确测试的是“完全被mask的数据不得参与”。
4. 对活跃行令Q=0，输出应为所有selected可见token的V平均；不是各页平均的再平均，尾页长度不同尤其关键。
5. 令所有可见V为同一常量，输出应接近该常量。
6. 改split数/merge树，结果应在预先讨论的误差界内一致；不要默认逐位相同。

指标至少包括：active行NaN/Inf计数、max absolute error、RMSE、逐query/head的NRMSE分布，以及全局NRMSE。可用：

$$
NRMSE_{row}=\frac{\|o-o_{ref}\|_2}{\max(\|o_{ref}\|_2,\tau)}.
$$

近零参考输出应以绝对误差为主，$\tau$ 必须说明，不能靠一个很小分母把有意义与无意义的相对误差混在一起。

**容差草案：** 可以从上游scale测试的 `atol=rtol=0.02` 和harness的全局相对L2<0.02出发，作为初始待挑战的回归门槛；同时增加逐行误差、非有限数和上述解析性质检查。这个数值不是已校准的通用精度承诺。先运行baseline对独立gold，检查极端输入的合理误差，再在实现候选前确定最终域与阈值；不要看候选结果后不断放宽阈值。

为何必须分层？令实际 $o=\sum p_jv_j$，小扰动的一阶表达为：

$$
\delta o\approx\sum_jp_j\delta v_j+
\sum_jp_j(v_j-o)\delta z_j.
$$

K/Q量化通过logit扰动改变权重，V量化则直接改变加权值；不能给所有输入统一套一个“FP8有效位数决定的最终相对误差”。此外，若merge权重精确、每个局部输出误差被同一个绝对界限制，最终是这些误差的凸组合，并不必然放大S倍；真正需要同时关注LSE误差和局部输出舍入。

**14．讨论点五：当前证据尚不能确定硬件瓶颈，应该怎样 profile？**

从第7节的请求AI，可以提出内存侧限制的假设。若以B300标称单卡dense BF16 2.25 PFLOP/s、HBM8 TB/s作理想参照，机器平衡点约281.25 FLOP/B，显著高于这里13～32 FLOP/B。规格依据 [HGX峰值口径](https://www.nvidia.com/en-au/data-center/hgx/) 和 [B300带宽](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory/latest/components.html)。

但**低AI不等于实际DRAM已饱和**。小batch可能只有16/64个partial CTA，每个CTA还经历查表、K加载、QK、softmax、V加载、PV，随后是merge。全卡算力低、DRAM带宽低、单CTA等待多可以同时成立。实际瓶颈应在这些候选中识别：

- HBM/L2的数据供给；物理页分布和缓存命中。
- 两级地址依赖与L1TEX/long-scoreboard延迟。
- short-K小矩阵的发射/operand staging，与softmax的归约/SFU路径。
- register spill、shared布局冲突或资源限制导致的低occupancy。
- split不足、空split、merge CTA粒度与启动/调度开销。
- CPU调度、分配、metadata、量化与Graph/PDL条件。

最小测量顺序应是：固定版本和环境；先跑合法域正确性；分别测partial、merge和完整wrapper；再采集硬件计数器；依据测得瓶颈选择候选；候选实现前确认验收方案。该顺序满足题目“测完再设计”的要求。

建议先采集下表，不把所有计数器堆进一次极慢的完整profile：

| 要回答的问题 | 证据/metric方向 | 判读边界 |
|---|---|---|
| 哪一步占主要时间 | kernel duration、CUDA事件与执行轨迹；partial/merge/metadata分别列 | 不能把FLOP占比当时间占比 |
| 是否达到DRAM限制 | `dram__bytes_read.sum`、`dram__bytes_write.sum`、DRAM throughput | 逻辑KV字节除时间不是实测HBM带宽 |
| cache是否重要 | L2 hit rate/bytes/throughput，冷热条件 | 反复相同KV/topk可能形成热缓存 |
| 并行工作是否不足 | grid、waves/SM、active/eligible warps、issue active | 发出CTA数、活跃SM覆盖、SM内occupancy是不同量 |
| matrix/SFU/归约的代价 | ComputeWorkloadAnalysis、tensor pipe与相关instruction统计 | Triton `tl.dot`的实际代码需查PTX/SASS |
| 地址与片上等待 | long/short scoreboard、barrier、shared wavefront/conflict | 单个stall名字不是唯一瓶颈证明 |
| 是否spill/资源不足 | registers/thread、SMEM/CTA、local load/store | 应与具体kernel版本和shape绑定 |

指标命名以目标GPU上 `ncu --query-metrics` 为准，解释方法见 [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)。应同时记录GPU型号/SM数、时钟/功耗状态、驱动/CUDA/Triton版本、是否Graph、PDL、热缓存、已预分配workspace，以及实际被调用的backend。

跨dtype比较时，相同的理论AI不能共享一个未经核实的compute roof：当前FP8 KV Triton会先解量化到BF16；CUTLASS若走不同矩阵路径，峰值口径与Q量化成本都可能不同。计算上界应与实际输入和累加类型匹配。

**15．harness 是起点，还不是完整性能/精度证据。**

现有 `harness/run.py` 的check包括常规、短序列尾块和d_q=2三类；bench扫描B1/4/8/16/32/64，固定TP1、BF16、seq_len8192、topk16，10次warmup后把100次调用放在同一CUDA-event区间取平均。每次调用分配output，底层再分配两个partial workspace。源码见 [run.py](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/harness/run.py:18)。

需要明确它的限制：

- 未覆盖FP8、TP4、CUTLASS以及真实变化的scale；全局err_ratio可掩盖个别query/head错误。
- shim强制PDL=false，和生产CUDA路径的条件PDL不同，见 [vllm_shim.py](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/harness/vllm_shim.py:15)。
- 相同输入/top-k重复运行会形成热缓存；TP1 B1/4/8/16选中的BF16 KV payload为4/16/32/64 MiB，不能自动假定每次都从HBM读取。
- synthetic物理页打乱是有价值的，但其页集合不包含真实共享前缀模式，也不包含indexer耗时；随机KV/top-k分布不是服务请求分布。
- 单次100调用平均没有重复间统计或P50/P95，也没有区分partial、merge、host调用、Graph/PDL和metadata。
- CUDA-event包住Python循环时，测到的GPU时间区间可能包含host供给不足产生的空隙；它不等于纯kernel计算时间，也不是完整host wall-clock。

扩展测量时应保留两种口径：预分配/固定输入地址下的kernel与device链路；公开接口或实际服务调用的端到端延迟。Graph和普通launch分开报告，冷/热/变化工作集分开报告，不能只给候选开启Graph或workspace复用。

现阶段不能从C1的B300 benchmark推断C2的时延，也不能把C1中ncu probe失败泛化成C2永久无法profile；这里只记录“本次没有C2 profile数据”，后续应在实际环境尝试并记录结果。

**16．CUTLASS crossover=16：已有事实与不能推出的结论。**

上游文件写明，kernel benchmark把TP1/TP4的交点放在16请求，并设置 `_MIN_CUTLASS_BATCH_SIZE=16`。这是“上游作者报告过这个经验结果”的源码证据，本快照没有附上原始benchmark表、计时范围、设备频率、各shape的profile，因此它不是本机已重现的结果。来源见 [CUTLASS门槛常量](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/vllm_msa_ref/msa_cutlass_sparse_decode.py:16)。

CUTLASS静态支持域不只有batch：

| 条件 | 固定版本要求 |
|---|---|
| backend | 显式选择cutlass，CUDA平台，capability family100 |
| KV dtype | `fp8`或`fp8_e4m3` |
| heads | $0<H_q\le64$、$0<H_{kv}\le4$，且整除 |
| 稀疏形状 | page128、topk16；底层接口契约D128 |
| query/计划 | $1\le d_q\le32$、$B\ge16$、$Bd_qH_q\le65536$ |

这里D128是接口常量/切分契约；`supports_cutlass_sparse_decode` 本身没有head_dim参数，不能说该guard已经检查了D。TP1的query-head-row上限相当于总query数1024，TP4相当于4096。B8,d_q32虽然有256个query，仍被B≥16的请求数门槛挡住，进一步说明门槛不是简单以总query行数决定的数学边界。代码见 [静态与动态guard](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/vllm_msa_ref/msa_cutlass_sparse_decode.py:199)。

实际forward还要求已准备的metadata存在，否则回退Triton。构造时的“CUTLASS”日志不足以证明每一批都走CUTLASS。性能对比必须检查真实dispatch，尤其B<16时如果只设置backend配置，仍可能测到Triton。

从已有代码能列出的工程成本包括：planner、CPU长度信息、page-indptr构造、以B/d_q/page stride/head形状等为key的计划缓存、保持Graph稳定的地址与运行时metadata更新。缓存命中后仍有一个metadata更新kernel；冷启动planner成本不能全算作每层每次稳态成本，runtime更新也不能完全忽略。见 [plan cache与prepare](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/vllm_msa_ref/msa_cutlass_sparse_decode.py:57)。

底层 `vllm.third_party.fmha_sm100` 实现没有包含在本快照。不能从wrapper编造CUTLASS的CTA grid、寄存器数、SMEM占用、流水级数或是否内部融合merge。要解释精确交点，后续需要固定完整上游版本并检查它的实际kernel与计划器。

可检验的交点假设包括：小B下专用路径的启动/元数据/布局成本占比大；大B提高全卡并行和摊销；Triton自身S会随shape变化；FP8 Q/K/V路径改变算术与访存成本；两条路径对热缓存、PDL和Graph的收益不同。第6节已排除“两个TP都在B16才有256个partial CTA”这个过度简化解释，但尚不能在剩余假设中选定主因。

比较时应报告相同数学输入域下的Triton与CUTLASS，同时列出Q量化是否已计入、metadata更新的调用频率与成本归属。若评估原生不同精度路径，还应单列质量/误差，避免把额外量化导致的速度当作无条件等价实现的加速。

**17．挑战路线(b)：“不值得做”也需要严格的收益上限。**

设当前实测端到端耗时为 $t_0$，在明确条件下得到任意候选不可突破的下界 $t_{lb}$，则：

$$
Speedup_{max}\le\frac{t_0}{t_{lb}},\qquad
\Delta t_{max}\le t_0-t_{lb}.
$$

roofline给的是吞吐上界，换成时间是下界。若冷数据必须读取给定KV、没有压缩/跨query复用，按标称8 TB/s，仅KV读取的乐观时间如下：

| 每卡形状 | B | BF16 KV读取假设时间 | FP8 KV读取假设时间 |
|---|---:|---:|---:|
| TP1 | 1 | 0.5243 µs | 0.2621 µs |
| TP1 | 16 | 8.3886 µs | 4.1943 µs |
| TP4 | 1 | 0.1311 µs | 0.0655 µs |
| TP4 | 16 | 2.0972 µs | 1.0486 µs |

这些是“全部payload由HBM提供且达到标称带宽”的假设计算，不是预测完整kernel延迟。热缓存可使HBM必读字节更少，小batch又可能远达不到额定带宽；若输入条件不满足，不能把表中时间当成有效下界。

因此没有 $t_0$ 与真实冷/热流量证据时，不能据此完成“不值得做”的论证。尤其一个过分乐观、很小的 $t_{lb}$ 会给出很宽松的加速上限，反而无法证明收益有限。

更有说服力的是限定优化切面。例如若profile显示merge和暴露launch仅占总时间p，候选只改变这部分，则理想消除后的Amdahl上限为 $1/(1-p)$；如果p很小，这个方向的收益确实受限。但这不否定其他方向，例如减少解量化成本或增加可调度工作。

工程成本应具体到维护范围：FP8格式与scale模式、Q量化、TP/投机形状、top-k与page布局、Graph/PDL、尾块/空请求、计划缓存、不同SM目标、数值验收与回退。wrapper可以证明存在这些集成面，但不能换算出未经依据的人日或费用。是否值得取决于目标请求分布、每token绝对延迟收益和维护资源，不能拿一个kernel百分比直接等同服务吞吐收益。

与C1的对照也有帮助：C1中时间状态依赖限制chunk并行；C2中split可以并行，但要保留softmax归约状态。C1的bf16风险来自长程反复状态量化；C2主要关注logit/概率/partial与FP8 scale的误差传播。两题都受小形状影响，却不应套用同一套瓶颈解释。

**18．下一步实验的顺序与本次CPU核对。**

这里给出研究顺序，具体实现选择仍等待profile：

| 次序 | 工作 | 应产出的证据 |
|---:|---|---|
| 1 | 在合法输入域验证现有Triton与独立reference | 正确性结果、baseline误差分布、harness不足 |
| 2 | B1/4/8/16 × TP1/TP4 × BF16/FP8，分项计时 | partial/merge/API耗时、真实S与CTA数、Graph/PDL/缓存条件 |
| 3 | 对代表性shape采集ncu/PTX/SASS | 带宽、依赖、issue、occupancy、spill与实际矩阵指令 |
| 4 | 固定验收域/容差并提交组间讨论 | 明确reference、有效行、scale/布局、失败标准 |
| 5 | 按测到的瓶颈选择一个切面 | 假设、预计节省项、反例与最低收益目标 |
| 6 | 正确性先过，再对完整路径测量 | 同口径性能、数值差异、支持域与回退域 |

题目要求把验收方案给其他组挑战；本文提供可用于讨论的草案，组间讨论尚未进行。

本次新增的 [CPU理论检查脚本](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/analysis/theory_checks.py) 仅依赖Python标准库，[运行结果](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c2_msa_decode/analysis/theory_checks.json) 验证了：

- 1/2/4/8/16/64分块（包含空split）的LSE合并与直接attention等价，double算术最大绝对差约 $3.33\times10^{-15}$。
- 物理KV页与scale一致重映射，输出最大差为0；selected块顺序置换，差约 $1.11\times10^{-16}$。
- 在同一个小型索引例子中，错误地用逻辑页下标读取物理scale，输出最大绝对差约0.4107，说明测试确实可以分辨该类错误。
- 两个scale移动反例、TP1/TP4 split/CTA表、FLOP/字节/AI与条件HBM时间计算。

这些结果是代数和测试构造的核对，不模拟GPU上的FP8编码/MMA舍入，不属于Triton baseline profile，也没有选择或实现挑战层kernel。

本文已能确定的理论结论是：GQA提供实际的数据复用；split输出必须按LSE质量合并；TMA可以服务查表后的规则页搬运；FP8 scale必须跟随物理token/head并放在正确的数学层；当前验收和benchmark脚手架都有明确覆盖边界。融合是否有利、哪个阶段主导时延、B16交点的主因，以及小batch专版是否值得发布，仍应由上述测量补上证据。
