---
title: 作业 2:Tensor Core & Pipeline
subtitle: Weiming HPC Training Camp $\times$ LCPU AI Infra Seminars · Session 3 / 4
---

# Preface {-}

> **个人学习解答版 · 更新于 2026-09-22**  
> 保留完整原 handout，每题后紧跟已讨论的解答、实验结果和证据。尚未完成的题目标为待完成；团队原题保留但不代填。GPU 记录来自 B300，未进行 5090 实测。历史验证以链接中的源码快照为准。助教参与的实现和修正已注明。


这次作业主要是 Session 3 的配套练习，内容围绕 Tensor Core 展开，其中 Module 4 还会用到 Session 4 介绍的 tiling、TMA 和 pipeline。完成作业时主要参考课程课件和 PTX ISA 文档。

题型沿用系列惯例，并新增 DERIVE：先手工推导，再写程序验证推导。共七种：
CONCEPT 概念题、DERIVE 推导题、MODIFY 改造题、DEBUG 修 bug 题、
EXPERIMENT 实验题、HANDS-ON 动手题、FROM-SCRATCH 从零实现题。标
Optional 的为选做。涉及代码的题在标题行右侧标出文件路径(相对
`assignment02/`)，题面细节以文件头注释为准；FROM-SCRATCH 题。

硬件与构建:主线环境是 B300，`cuda/Makefile` 默认
`ARCH=100f`；M0、M1 也可以在 5090 上完成(`ARCH=120a make ...`)；
M2 的判测纯 host，无卡可判；M3、M4、M5 需要 B300。本作业统一用显式
`-gencode` 而不用 `-arch=sm_XXXa` 简写，原因见 `assignment02/README.md`，Makefile 已经配好。

实验纪律:所有性能数字都在自己占满的卡上测；测前测后
`nvidia-smi --query-compute-apps=pid，name --format=csv` 查有没有别的
进程占卡；可能 hang 的程序套 `timeout -k 5 <秒数>`(判测脚本已带)。

AI政策:必做题沿用仓库根目录 `CLAUDE.md`——AI 可以帮你理解、
review，不能替你实现；团队题不设限制。详见 `assignment02/README.md`。

## Content {-}

| Module | 主题 | 对应课件 | 代码位置 |
|---|---|---|---|
| 0 | 环境与峰值 | P1(S008--S021) | `cuda/m0_env/` |
| 1 | sm80:fragment 与 mma.sync | P2.2(S025--S041) | `cuda/m1_sm80/` |
| 2 | smem 供数:descriptor 与 swizzle | P2.3(S042--S070) | `cuda/m2_smem/` |
| 3 | sm100:tcgen05 | P2.4(S071--S093) | `cuda/m3_tcgen05/` |
| 4 | 完整 GEMM 四步走 | S084 + Session 4 | `cuda/m4_gemm/` |
| 5 | 低精度与 block scaling | P3(S094--S111) | `cuda/m5_lowprec/`、`kernels/` |
| 6 | TileLang 对照 | 收尾(S112) | (复用 assignment01) |
| Team | FlashKDA / MSA decode | — | `team/` |

# 环境与峰值

先确认工具链和卡都能发出 tensor core 指令：编译一个最小tensor core程序，测试卡的峰值性能，后面就可以用它来计算各个实现的性能达成率。

::: reading
Session3课件S008-S021；PTX ISA 的 mma 指令一节(形状与 dtype 表)；你手里
每块卡的官方 datasheet(或 whitepaper)。
:::

### 0.1 {.prob type=HANDS-ON file=cuda/m0_env/01_first_mma.cu}

编译并运行最小 Tensor Core 程序，它使用一个 warp 执行一条 m16n8k16 的 mma.sync 指令，并与 CPU 计算结果进行比较。

```
cd assignment02/cuda
make run/m0_env/01_first_mma
```

在你能使用的 GPU 上分别运行该程序（5090 使用 ARCH=120a make ...，B300 使用默认配置）。尝试使用不匹配的 ARCH 编译运行一次，记录现象，并结合 assignment01 Module 8 中 fatbin/JIT 的内容解释原因。可使用 make ptx/m0_env/01_first_mma 查看生成的 PTX。

#### 0.1 解答与实验记录

B300 上，`ARCH=100f` 编译及运行成功，四个输出均为 2；`ARCH=120a` 编译成功，但运行报 `cudaErrorNoKernelImageForDevice`。编译器可以在没有目标 GPU 的机器上生成 cubin，是否能执行由目标架构兼容性决定。B300 的 compute capability 为 10.3，不能运行面向 12.0 架构专属特性的 120a cubin。

这次两个二进制的 `cuobjdump --list-ptx` 均没有列出嵌入 PTX；不能假定 driver 会自动找到另外导出的 `.ptx` 文件并回退。Driver JIT 需要程序提供可兼容的 PTX。单独导出的 PTX 为 `.version 9.1`、`.target sm_100f`，含 `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`。

证据：[0.1 编译、运行与 PTX 记录](../../experiments/results/m0-20260913T065647046705Z/OBSERVATIONS.md)。



### 0.2 {.prob type=DERIVE}

推导你所使用 GPU 的 Tensor Core 理论峰值。参考课上 A100 的推导方法（S018--S019），分别计算 5090 和 B300 的 bf16 峰值，并根据 dtype 宽度关系估算 fp8 / fp4 峰值。

开始计算前先明确采用的口径，包括 dense 或 sparse、boost 或 base 频率、FMA 是否计作 2 FLOP 等。完成推导后，再与 datasheet 中的官方数据进行对照；如果结果存在差异，需要说明差异来自哪一项口径。

| 量 | RTX 5090 | B300 SXM |
|---|---|---|
| bf16 FLOP/cycle/SM | 512（依据白皮书峰值、170 SM、2407MHz反算） | 8192（Blackwell tcgen05 满速 BF16 吞吐模型） |
| bf16 峰值(TFLOPS) | 209.5 | 2250 |
| fp8 峰值(TFLOPS) | 位宽估计419；白皮书FP32累加419，FP16累加838；新kind路径需另分口径，见下文 | 位宽估计4500；官方dense4500 |
| fp4 峰值(TFLOPS) | 简单位宽估计838；官方FP32累加dense1676 | 简单位宽估计9000；官方dense13500 |
| datasheet 对照值与口径差异 | BF16 209.5/419、FP8 FP32累加419/838、FP4 1676/3352分别为dense/sparse；不可把3352 AI TOPS当BF16峰值 | HGX为8卡总量：BF16 36PF sparse、FP8 72PF sparse，单卡dense各除以16；FP4明确列144PF sparse/108PF dense，单卡dense除以8 |
| HBM/GDDR 带宽(GB/s) | 1792，GDDR7 | 8000，HBM3e，规格上限 |
| 机器平衡点(FLOP/byte，bf16) | 209.5×1000/1792 ≈ 116.91 | 2250×1000/8000 = 281.25 |

根据 bf16 峰值和显存带宽计算机器平衡点（FLOP/byte），并与单条 mma 的计算强度（S016，m16n8k16 fp16 为 3.2 FLOP/byte）比较。思考两者之间的差距意味着什么，以及为什么后续 M2--M4 需要从数据供给路径入手优化。

#### 0.2 解答与实验记录

统一口径：单 GPU、dense、FMA=2 FLOP；BF16 输入与 FP32 累加。频率用于标称模型，不把运行时动态时钟或指令依赖延迟当成固定吞吐。TFLOPS、GB/s 均使用十进制单位。2026-09-22 按本人请求由助教查阅官方资料补表。

| 量 | RTX 5090 | B300 SXM |
|---|---|---|
| bf16 FLOP/cycle/SM | 512（依据白皮书峰值、170 SM、2407MHz反算） | 8192（Blackwell tcgen05 满速 BF16 吞吐模型） |
| bf16 峰值(TFLOPS) | 209.5 | 2250 |
| fp8 峰值(TFLOPS) | 位宽估计419；白皮书FP32累加419，FP16累加838；新kind路径需另分口径，见下文 | 位宽估计4500；官方dense4500 |
| fp4 峰值(TFLOPS) | 简单位宽估计838；官方FP32累加dense1676 | 简单位宽估计9000；官方dense13500 |
| datasheet 对照值与口径差异 | BF16 209.5/419、FP8 FP32累加419/838、FP4 1676/3352分别为dense/sparse；不可把3352 AI TOPS当BF16峰值 | HGX为8卡总量：BF16 36PF sparse、FP8 72PF sparse，单卡dense各除以16；FP4明确列144PF sparse/108PF dense，单卡dense除以8 |
| HBM/GDDR 带宽(GB/s) | 1792，GDDR7 | 8000，HBM3e，规格上限 |
| 机器平衡点(FLOP/byte，bf16) | 209.5×1000/1792 ≈ 116.91 | 2250×1000/8000 = 281.25 |

推导采用 `P = SM数 × 每SM每周期FLOP × 频率`。5090 白皮书列170 SM、每SM四个Tensor Core和2407MHz；以512 FLOP/cycle/SM计算为 `170×512×2.407/1000=209.50528 TFLOPS`。这里512来自规格反算，不冒充独立MMA吞吐或延迟微基准；官网2.41GHz是显示精度较粗的值。[RTX Blackwell白皮书，Table 3及脚注](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)

B300 使用8192 FLOP/cycle/SM的BF16模型；项目已留存的B300设备探测为148 SM。若以官方2250TFLOPS标称值校准，等效频率为 `2250×1000/(148×8192)≈1.85580 GHz`。这是**从官方峰值反推的模型频率，不是读取到的boost规格，也不是独立验证**。取1.86GHz时得到2255.09TFLOPS，与官方舍入值接近。实际实验需另记录动态时钟。CUTLASS说明Blackwell `tcgen05.kind::f16` 相对Hopper吞吐加倍；本表使用相应8192模型，不从一条MMA有多少在飞或延迟多少周期直接推出它。[CUTLASS架构说明](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html)

官方B300数值按HGX B300 SXM口径取值：FP4的dense不是sparse的一半，该表已分别列出108PF/144PF（八卡）。Ultra的dense NVFP4增强使其不同于简单的位宽减半估计。不要把最高配置Blackwell Ultra/GB300宣传中的15PF直接混入本表的B300 SXM 13.5PF。[HGX规格和脚注](https://www.nvidia.com/en-us/data-center/hgx/)、[Blackwell Ultra架构说明](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/)

5090 的 FP8 还必须区分指令路径：白皮书的FP32累加行列419TFLOPS；CUTLASS又明确说明SM120新增 `mma.sync.aligned.kind::f8f6f4` / block-scale路径对FP32 accumulator具有相对Ada FP8的2倍吞吐。不能把419作为所有SM120 FP8指令的统一上限；若后续使用新kind路径，应单独使用其吞吐口径并实测。FP4的官方1676也说明只从BF16按位宽推成四倍会低估新低精度路径。[CUTLASS SM120说明](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html#blackwell-sm120-gemms)

带宽独立核对：[5090官方发布规格](https://www.nvidia.com/en-gb/geforce/news/rtx-50-series-graphics-cards-gpu-laptop-announcements/)、[HGX每GPU带宽表](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory/latest/components.html)。

与单条MMA的3.2 FLOP/byte相比，BF16机器平衡点分别约为其36.5倍和87.9倍。这说明若每次MMA都从显存重新取全部输入，供数无法匹配算力；GEMM需要跨输出复用输入，并通过tiling、shared、寄存器/TMEM与流水提高有效利用率。但3.2是指令操作数口径，不等于整个kernel的实际HBM算术强度，不能直接代入kernel roofline。


### 0.3 {.prob type=CONCEPT}

判断下列说法是否正确，并给出一句理由。

(a) 一条 mma 的计算强度，分子是 $2MNK$，分母按 A、B 读入与 D 写回
的字节总和计(S016 的口径)。

(b) mma.sync 是 warp 级协作指令:32 个 lane 各持 fragment 的一部分，
要求全 warp 一致地执行这条指令；有 lane 发散时行为未定义。

(c) 增大 mma 的形状 M/N/K 能提高单条指令的计算强度，而且没有代价，
所以指令形状越大越好。

(d) 只要单条 mma 的计算强度低于机器平衡点，GEMM kernel 就不可能逼近
计算峰值。

#### 0.3 解答与实验记录

依题目顺序为：**对、对、错、错**。

- 一次矩阵乘的运算量按 FMA=2 FLOPs 计为 `2MNK`；题设只计 A、B 的读取及 D 的写回时，字节数为各矩阵元素数乘对应元素字节数。
- `mma.sync` 要求 warp 中参与线程一致执行相同指令。进入 MMA 前已经重新汇合的分支，与让部分 lane 跳过 MMA 不同。
- 增大 MMA 形状并不保证提高实际利用率，还会增加准备操作数、寄存器等资源压力。
- 复用能减少重复搬运、提高算术强度；pipeline 可以重叠供数和计算、隐藏等待，但仅仅重叠并不改变运算量与数据量之比。

讨论原记录：[0.3](../../experiments/m0_concepts.md)。



# sm80:fragment 与 mma.sync

课上对 m16n8k16 fp16 推过 fragment 公式、手搓过单 tile mma
(C03--C07)。现在你手推一遍这个 m16n8k32 fp8 shape，
再看 ldmatrix 到底发挥了什么作用。

::: reading
课件 S025--S041、C03--C08；PTX ISA 的 "Matrix Fragments for
mma.m16n8k32" 一节与 "Warp-level matrix load instruction: ldmatrix"
一节；fp8 转换用 `cuda_fp8.h`(`__nv_fp8_e4m3`)。
:::

### 1.1 {.prob type=DERIVE file=cuda/m1_sm80/01_fragment_map.cu}

根据 PTX 文档中的 fragment 布局，推导
`mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`
对应的 A、B fragment 映射公式，并完成文件中的四个函数。
判测使用纯 host 的 32-lane 真值表比对，无需 GPU 即可完成：

```
cd assignment02/cuda
make run/m1_sm80/01_fragment_map
```

附加问题（写入报告）：
A 的同一个 b32 寄存器中的 4 个 fp8 元素沿矩阵哪个方向相邻？
这个布局对 1.4 中使用 ldmatrix load 有什么影响？

#### 1.1 解答与实验记录

这里对应 `mma.m16n8k32` 的 FP8 布局，不能套用元素数相同但布局不同的另一幅图。令 lane 为 0–31，A 的 i 为 0–15，B 的 i 为 0–7，已实现的映射为：

```cpp
A_row = (lane >> 2) | ((i & 4) << 1);
A_col = ((lane & 3) << 2) | (i & 3) | ((i & 8) << 1);
B_row = (i & 3) | ((lane & 3) << 2) | ((i & 4) << 2);
B_col = lane >> 2;
```

每个 b32 寄存器装四个 FP8 元素。A 中这四个元素沿 K（列）连续，B 中沿 K（行）连续。因此 `lane=0,i=4` 对应 A(8,0)，不是 A(0,4)。`ldmatrix.b16` 可以把两个 FP8 字节当成一个 16-bit 单元搬运，不做浮点格式转换；目的寄存器仍为 b32。

本人实现的 A 512 项、B 256 项 host 真值检查通过：[原始记录](../../experiments/results/m1-fragment-20260915T120220Z/results.json)。2026-09-22 对当前源码再次检查仍 PASS：[本次 host 复核](../../experiments/results/host-review-20260922T033125Z/results.json)。



### 1.2 {.prob type=DEBUG file=cuda/m1_sm80/02_bug_fragment.cu}

这个程序发一条 m16n8k16 fp16 mma，判测会 FAIL。先运行一遍，后改动:

(a) 描述症状：D 的哪些位置错、错成了什么(和对的部分是什么关系)；

(b) 修好它，并解释错的是哪个 fragment 的哪部分映射，为什么恰好产生(a)的症状。

```
cd assignment02/cuda
make run/m1_sm80/02_bug_fragment
```

#### 1.2 解答与实验记录

现象是 D 的 M 方向重复。追溯到 A 的下半部分行数据错误地重复使用了上半部分：需要来自下方 8 行的 `a2/a3/a6/a7` 没有正确取行。本人定位并修正后，固定输入由 **FAIL 59/128 → PASS 128/128**。不匹配数不是整齐的 64，是因为部分数值碰巧相等。

证据：[修改和 B300 对拍](../../experiments/results/m1-bug-20260915T130436Z/OBSERVATIONS.md)。



::: {.capstone title="prob 1.3(FROM-SCRATCH):手写单 tile fp8 mma"}

在 `cuda/m1_sm80/03_mma_fp8.cu` 中从零实现一个单 tile 的 fp8 mma，不提供代码骨架。使用 `m16n8k32` e4m3 mma、f32 累加，并手动完成 fragment 装载（本题不使用 ldmatrix），最后与 CPU 参考结果进行严格相等比较。

要求如下：

输入使用小整数，保证转换为 e4m3 后可以精确表示；

程序接受一个 seed 参数，例如 `./prog 123`；

输出必须以 `PASS` 或 `MISMATCH` / `FAIL` 开头；

fp8 与 float 的互转直接使用 `cuda_fp8.h`。

1.1 中推导的 fragment 映射公式会直接用在这里。映射只要有一处错误，随机数据的判测就无法通过。

判测脚本会使用五个 seed，全部通过才算完成：

```
cd assignment02/cuda/m1_sm80
./judge_mma_fp8.sh 03_mma_fp8.cu
```

:::



#### 1.3 解答与实验记录

本人编写 kernel 主体，助教按请求补 main 并协助修正 D 映射、字节打包和公共头文件复用。FP8 类型来自 `cuda_fp8.h`；小整数测试输入能精确表示，测试选用的小范围不是 FP8 的完整动态范围。

D 的每线程四个元素映射为 `row=(lane>>2)+(i>=2?8:0)`、`col=2*(lane&3)+(i&1)`。在 B300 上使用课程五个 seed（1、7、42、1234、99999），每次 128 个结果严格对拍全部通过。

证据：[1.3 记录](../../experiments/results/m1-fp8-20260916T010107Z/OBSERVATIONS.md)。

### 1.4 {.prob type=MODIFY file=cuda/m1_sm80/04_ldmatrix.cu}

在给定骨架中保留 1.3 的手工装载方式，并另外实现一条使用 `ldmatrix` 的装载路径。两种实现需要共存，并分别通过判测。

根据 PTX 文档选择合适的 `ldmatrix` 变体（`.x1/.x2/.x4`，以及是否使用 `.trans`）。B 矩阵在 shared memory 中的布局还需要满足 `ldmatrix` 的 16 byte 行地址要求，具体约束见文件头说明。

两条路径都通过判测后，使用 `make ptx/m1_sm80/04_ldmatrix` 或 `nvdisasm` 查看生成的指令，分别统计 `smem → fragment` 阶段的装载指令和地址计算指令数量，并回答：

(a) `ldmatrix` 省掉了手工装载中的哪些工作？

(b) 为什么这些工作在手工装载路径中无法避免？

#### 1.4 解答与实验记录

按本人请求由助教完成教学实现。对照 MMA 所需的寄存器布局和 ldmatrix 的搬运布局，构造 shared 地址；不能仅由 MMA 的 `.row.col` 推出 B 必须使用 `.trans`。

已验证版本中 A 使用 `.m8n8.x4.b16`，四块按左上、左下、右上、右下排列。每个 lane 提供的字节地址为 `A+(lane&15)*32+(lane>>4)*16`。B 先暂存成 `B[n][k]`，让 K 连续，再使用不带 `.trans` 的 `.x2`；提供地址的前 16 个 lane 使用 `B+(lane&7)*32+((lane>>3)&1)*16`。提供行地址的 lane 与最终获得片段的 lane 是两种角色。

`.trans.b16` 转置的是 16-bit 单元，不能当作逐个 FP8 字节的转置。B 的手工路径和 ldmatrix 路径最终都必须得到每个寄存器内沿 K 连续的四个 FP8。

B300 三个 seed（1、7、42），两条路径均 128/128 PASS。对保存的 PTX，仅统计 smem→fragment 这一段：

| 类别 | 手工路径 | ldmatrix 路径 |
|---|---:|---:|
| 装载 | 12 | 2 |
| 地址算术 | 12 | 10 |
| 基址搬运 | 2 | 2 |
| 打包 | 12 | 0 |
| 合计 | 38 | 14 |

不包含 global→shared、B 暂存布局准备等；这是 PTX 静态统计，不是 SASS 数或速度提升。后来编辑过的当前源码不能仅凭该历史结果视为重测通过。

证据：[已验证源码和统计口径](../../experiments/results/m1-ldmatrix-20260920T075433Z/OBSERVATIONS.md)。



### 1.5 {.prob type=EXPERIMENT file=cuda/m1_sm80/05_ldsm_stride.cu}

观察行跨度对 `ldmatrix` 的影响。对同一个 16×16 fp16 tile，分别使用
32 B、64 B、128 B 和 128+16 B（padding）四种行跨度。

先根据课上介绍的 bank 模型，预测四种情况下执行一次 `ldmatrix`
所需的 wavefront 数及其比例，再运行程序并使用 Nsight Compute 测量：

```bash
cd assignment02/cuda
make run/m1_sm80/05_ldsm_stride
ncu --metrics l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum ./bin/m1_sm80/05_ldsm_stride
```

| 档位 | 预测 wavefront 比 | 实测 wavefront | 实测 conflict | 平均 cycle |
|---|---|---|---|---|
| 32 B | | | | |
| 64 B | | | | |
| 128 B | | | | |
| 128+16 B | | | | |

比较预测与实测结果：哪一种行跨度使 wavefront 数增加到 4 倍？
wavefront 的比例应与 bank 模型一致，但实际耗时的差距通常没有这么大。
结合 8 个 warp 的占用情况，解释为什么 wavefront 增加 4 倍并不会使总耗时也增加 4 倍。

#### 1.5 解答与实验记录

使用 `bank=(byte_address/4)%32` 推导。对一个 8×8 b16 子矩阵，stride 32/64/128/144 B 分别需要 2/4/8/1 个 wavefront；`.x4` 对应 8/16/32/4。本人先预测，再运行 B300 NCU 验证：

| stride（B） | kernel wavefront | kernel conflict | 每条实际 warp LDSM wavefront | 每条实际 warp LDSM conflict | 程序打印 cycle |
|---|---:|---:|---:|---:|---:|
| 32 | 16384 | 8192 | 8 | 4 | 9.72 |
| 64 | 32768 | 24576 | 16 | 12 | 10.75 |
| 128 | 65536 | 57344 | 32 | 28 | 16.06 |
| 144 | 8192 | 0 | 4 | 0 | 9.23 |

归一化必须看实际 SASS：PTX 中展开的重复装载被合并，机器码每个循环体只剩 2 条 LDSM，循环 128 次、8 个 warp，共 2048 条 warp LDSM。不能按源码 `8*4096` 除计数器。

wavefront 比为 2:4:8:1，但原程序 128B/32B 耗时比约 1.65。循环还有 LOP3、循环控制、依赖及多个 warp 交错执行；wavefront 工作量不等于整个循环周期。打印 cycle 是 `(t1-t0)/4096`，不是单条实际 LDSM 延迟，不能据此量化延迟隐藏。普通计时与 NCU replay 计时分开。

证据：[原始 NCU、SASS 与归因限制](../../experiments/results/m1-stride-20260920T090419Z/OBSERVATIONS.md)。



::: lookback

1.3--1.5 关注的是同一个问题：怎样把数据从 shared memory 装入
Tensor Core 的 fragment。1.3 手动完成每个 lane 的装载，1.4 使用
`ldmatrix` 简化这一步，而 1.5 说明即使使用 `ldmatrix`，shared memory
的 bank conflict 仍然会影响实际效率。

后面的模块会继续沿着这条数据供给路径展开：M2 中使用 descriptor
描述布局，M4 中进一步使用 TMA 和 pipeline 完成数据搬运。

:::

# smem 供数:descriptor 与 swizzle

sm90 起，tensor core 从 smem 取数不再经过 fragment 装载指令，而是
读一个 64 位 descriptor 按 canonical 布局自取。descriptor 与 swizzle
的格式在 sm100 的 tcgen05 上原样沿用，本模块推的每个公式都是 M3、
M4 的直接组件。

::: reading
课件 S042--S070(proxy:S051；core matrix 与 LBO/SBO:S057--S061；
swizzle:S062--S065)；PTX ISA 的 "Asynchronous Warpgroup MMA Shared
Memory Layout" 与 swizzling 小节。
:::

### 2.1 {.prob type=CONCEPT}

(a) 一个 warpgroup 使用 wgmma 读取刚写入 shared memory 的数据。将下面六个操作排成正确顺序，并说明每一步用于避免哪两个参与者之间的哪种乱序：

`wgmma.mma_async` / `st.shared` / `wgmma.commit_group` /
`fence.proxy.async` / `wgmma.fence` / `wgmma.wait_group`

(b) 判断下列说法是否正确，并给出一句理由。

1. `fence.proxy.async` 是 wgmma 专属的指令，TMA 与 tcgen05 的场景不需要它。
2. `wgmma.commit_group` 会阻塞，直到它之前发射的 wgmma 全部完成。
3. 不加 `fence.proxy.async` 时，wgmma 可能读到 shared memory 中的旧值，因为 `st.shared` 的写经过 generic proxy，而 wgmma 的读经过 async proxy。

#### 2.1 解答与实验记录

本题排序为：

```text
st.shared → fence.proxy.async → wgmma.fence
          → wgmma.mma_async → wgmma.commit_group → wgmma.wait_group
```

`fence.proxy.async` 建立 generic proxy 写 shared 与 async proxy 读取之间的顺序；`wgmma.fence` 处理此前寄存器访问与随后 WGMMA 寄存器访问的排序，不能替代前者。`commit_group` 将此前未提交的操作归组，不代表执行完成；读取结果前需要相应的完成等待，`wait_group 0` 等待全部已提交组。跨线程生产和消费还要满足对应的线程同步，不能将这一简化顺序当成完整的多线程协议。

三个判断为：**错、错、对**。proxy fence 并非 WGMMA 专属；关键是是否跨 proxy、是否需要建立可见性，不是所有 shared 读取都无条件加同一种 fence。




### 2.2 {.prob type=DERIVE file=cuda/m2_smem/02_descriptor.cu}

根据文件头给出的位域，实现 SM100 的 64 位 smem matrix descriptor 编码函数，并分别对下面三种情况推导 LBO、SBO 和 layout：

- K-major，无 swizzle；
- K-major，128B swizzle；
- MN-major，128B swizzle。

判测为纯 host，无需 GPU。测试使用的描述符真值已经在 B300 上通过实际 tcgen05 指令验证，3.2 中也会使用这三组描述符：

```
cd assignment02/cuda
make run/m2_smem/02_descriptor
```

场景 2 和场景 3 最终得到的 descriptor 相同。在报告中回答：MN-major 与 K-major 的区别体现在哪里？

#### 2.2 解答与实验记录

已讨论的 `(LBO字节数,SBO字节数,layout)` 为：无 swizzle 的 K-major `(128,1024,0)`；128B swizzle 的 K-major 和本题 MN-major 均为 `(0,1024,2)`。这里的 0 是本题特定布局的结果，不应推广成所有 swizzle 布局都忽略 LBO。地址及 stride 按描述符要求的 16B 单位编码，不能直接把字节数写入位域。

K-major/MN-major 的选择在 MMA instruction descriptor 的相关 major 位中；对于本题 `.kind::f16`，A、B 对应 bit 15、16。两种 major 可以具有相同的 smem 布局字段；不同起始地址仍会造成整个 64-bit descriptor 不同。

**当前源码未全部通过。** 2026-09-22 host 复核：场景 1 FAIL，LBO 编码得到 `0x1`；场景 2、3 PASS。上述是已经讨论的推导，不是声称当前实现已修复。本次整理没有改动该源码。

证据：[当前 host 输出](../../experiments/results/host-review-20260922T033125Z/results.json)。




### 2.3 {.prob type=FROM-SCRATCH file=cuda/m2_smem/03_swizzle.cu}

实现 128B、64B 和 32B 三种 swizzle 模式的地址映射函数，将逻辑坐标映射到 atom 内的物理字节偏移。

课件 S064 已经推导了 128B swizzle 的地址位异或关系；64B 和 32B 两种模式需要根据 PTX ISA 的 swizzling 一节自行推导。

判测分为两部分：首先检查映射是否为双射，然后检查列访问是否存在 bank conflict。单纯使用 padding 或恒等映射虽然可能通过第一项，但无法通过第二项。判测同样为纯 host：

```
cd assignment02/cuda
make run/m2_smem/03_swizzle
```

本题实现的 `swizzle_128B` 会在 3.2 中直接用于 shared memory staging，并接受实际硬件上的 GEMM 判测。若 3.2 或 4.1 出现问题，可以先运行本题的判测，排除 swizzle 布局错误。

#### 2.3 解答与实验记录

令 `g=colByte>>4`、`b=colByte&15`，当前代码实现：

```text
128B: row*128 + ((row    ^g)*16) + b
 64B: row*64  + (((row&3)^g)*16) + b
 32B: row*32  + (((row&1)^g)*16) + b
```

本次 host 三种模式全部 PASS，证明满足当前题目检查器的双射和访问条件。但 64/32B 的课程行坐标映射与按连续字节地址解释的 PTX 硬件 swizzle 不能直接等同。

对按 `x=row*W+colByte` 连续排列、相应对齐基址下的 atom，PTX 字节地址异或关系可写为 `x ^ (((x>>7)&(W/16-1))<<4)`。因此 W=64/32 时参与异或的是 `row>>1`/`row>>2` 的低位，不是直接取 row 的低位。例如 64B 模式 `(row=1,colByte=0)`，当前课程函数返回 80，而上述硬件字节映射返回 64。

128B 模式两者一致，已在 3.2 的硬件 GEMM 中使用并通过；64/32B 目前只有课程 host 检查，没有硬件消费验证。**不把三项 host PASS 写成三种 PTX 硬件布局均已验证。**

证据：[当前 host 输出](../../experiments/results/host-review-20260922T033125Z/results.json)。




::: lookback

课件 C12--C14 给出了使用 wgmma 实现单 tile 的完整示例，本作业不单独设置对应题目。wgmma 是 sm_90a 专属指令，5090（sm120）和 B300（sm100）均无法运行；有兴趣的同学可以在集群 H 卡上自行尝试。

本模块重点保留 wgmma 路径中可以继续使用的两个部分：descriptor 与 swizzle。2.2、2.3 先分别完成它们的推导与判测，后面的 M3 会在实际硬件上继续使用。

:::

# sm100：tcgen05

在 sm100 上，tcgen05 将累加器放入 TMEM，并使用 mbarrier 处理异步完成通知。本模块会先在 B300 上完成一个单 tile GEMM，再通过后续实验理解 TMEM、mbarrier 和 2-CTA MMA 的使用方式。

::: reading
课件 S071--S093（TMEM：S073--S075；mma 与 idesc：S076--S079；
mbarrier：S080--S084；ld 与同步：S085--S086；七步流程：S087/F27；
2-CTA：S091）；PTX ISA 的 tcgen05 一族小节（alloc / mma / commit /
ld / fence）与 mbarrier 一节。
:::


### 3.1 {.prob type=CONCEPT}

判断下列说法是否正确，并给出一句理由。其中 (d) 需要写出计算过程。

(a) `tcgen05.ld` 读取 TMEM 时，每个 warp 只能读取自己对应的 32 条 lane，warp 之间不能互相读取。

(b) 与 `mma.sync` 由 warp 协作、wgmma 由 warpgroup 协作不同，`tcgen05.mma` 由单个线程发射，随后由硬件异步执行。

(c) TMEM 中的累加结果可以直接通过 TMA 搬回 global memory，不需要经过寄存器。

(d) TMEM 每个 SM 包含 128 lane × 512 column × 4 B；一个 m128n256 的 f32 accumulator 恰好占用其中一半。

(e) `tcgen05.commit` 会阻塞直到之前发射的 mma 全部完成，因此 commit 返回后即可安全读取 TMEM。

#### 3.1 解答与实验记录

依次为：**对、对、错、对、错**。

- (a) 在本题 warpgroup 的读取布局中，各 warp 对应自己的 32 条 TMEM lane；不能据此说整个 CTA 中永远只有一个 warp 能访问某一段。
- (b) MMA 由一个线程发射并异步执行；alloc/ld 等指令各有自己的协作范围，不能照搬单线程发射规则。
- (c) 结果通过 `tcgen05.ld` 到寄存器，再写回 global，不能直接用 TMA 从 TMEM 写回。
- (d) 按题面容量：`128*512*4=256 KiB`，`128*256*4=128 KiB`，占一半。申请按列计，一列包含 128 个 32-bit lane 元素，不是一列总共仅 4 B。
- (e) commit 发出完成通知请求，不阻塞到 MMA 完成；需要 mbarrier 等待及相应排序后再读取。




::: {.capstone title="prob 3.2(FROM-SCRATCH):tcgen05 单 tile GEMM" file=cuda/m3_tcgen05/02_single_tile.cu}

从零实现一个 tcgen05 单 tile GEMM：计算 m128n64k64 的 bf16 矩阵乘，使用 f32 累加、`cta_group::1`，并由一个 128 线程的 block 完成。

数据按照下面的路径流动：

`global → smem（K-major + 128B swizzle）→ tcgen05.mma → TMEM → tcgen05.ld → global`

代码骨架只保留课件 F27 中的七步流程注释。descriptor 使用 2.2 中实现的编码方式，shared memory staging 使用 2.3 中的 `swizzle_128B`。TMEM alloc、mbarrier、idesc 以及 mma / ld 的具体写法需要根据 PTX ISA 完成，相关语义可参考课件 C15--C21。

`fence.proxy.async` 的位置以及 `tcgen05.ld` 的 warp 可见范围已经在文件头中给出。

判测：

```
cd assignment02/cuda
make run/m3_tcgen05/02_single_tile
cd m3_tcgen05 && ./judge_tile.sh
```

在正确版本通过后，故意去掉 `fence.proxy.async` 再运行一次，记录出现的现象并写入报告。结合 2.1(a) 中的排序问题解释原因。

:::




#### 3.2 解答与实验记录

按本人明确请求由助教补全教学实现，然后逐段学习。计算 BF16 `128×64×64`，FP32 累加。A/B 以 K-major、128B swizzle 暂存，占 16/8 KiB；申请 64 列 TMEM，即 32 KiB。

执行流程：初始化 shared 中的 mbarrier 和 TMEM 地址槽；由一个完整 warp 分配 TMEM；协作 staging，并建立 proxy 可见性和线程同步；一个线程发射四次 K=16 的 MMA（覆盖 K=0…63），第一次不累加旧 TMEM、后续累加；commit 到 mbarrier；等待 phase 0 完成并执行相应 fence；各 warp 通过 `tcgen05.ld` 读回，等待 load 完成再使用寄存器；所有消费者结束后释放 TMEM。

`alloc` 的 `[dst]` 是用于接收 TMEM 地址的 **shared memory 地址**；写回的值才是 TMEM 地址。inline asm 的 `"r"`/`"l"` 描述 32/64-bit 寄存器操作数，`"=r"` 是输出约束。末尾 `"memory"` 是告知编译器内存可能被读写的 clobber，限制编译器重排；它本身不发出 GPU fence，不能代替线程同步或异步完成等待。

官方五 seed 均通过，8192 个输出逐项匹配；Compute Sanitizer memcheck、synccheck 均为 0 errors。证据：[3.2 保存结果](../../experiments/results/m3-tile-20260921T143738Z)。**题目要求的删除 proxy fence 观察实验还没有做，本题不能标为全部完成。**

### 3.3 {.prob type=DEBUG file=cuda/m3_tcgen05/03_bug_mbarrier.cu}

这个程序连续发射多轮 mma，并在每一轮后读取结果，用来模拟 M4 中沿 K 维流式计算的过程。当前程序的 mbarrier 使用存在错误，判测带有超时机制，因此程序挂死本身也是需要观察的现象。

先运行：

```
cd assignment02/cuda
make bin/m3_tcgen05/03_bug_mbarrier
cd m3_tcgen05 && ./judge_mbar.sh
```

(a) 分别记录 `rounds = 1`、`2`、`4` 时的运行结果以及问题出现的条件。

(b) 修复程序，并画出修改前后 mbarrier 的状态变化，包括 phase 和 arrival count。指出错误版本在哪一次等待中过早或过晚放行，并解释为什么会产生 (a) 中观察到的现象。

提示：注意 `tcgen05.ld` 与尚未完成的 mma 之间的关系。

#### 3.3 解答与实验记录

原版用固定 `mbar_wait(mbar_u32,0)`：两个 seed（42、7）均 rounds=1 PASS，rounds=2/4 在 5 秒后超时，退出码 124。本人改为 `mbar_wait(mbar_u32,round&1)` 后，六个用例全部 PASS。每次 correctness 使用 `timeout -k 2s 5s`，无自动重试。

mbarrier 初始化 expected arrival=1。MMA 完成通知使 pending 从 1 降到 0，完成当前 phase，然后重新装载计数并翻转 parity。四个 warp 等待是消费者等待，不是四次 arrival。

```text
初始：phase 0，pending 1
  round 0 MMA完成通知 → phase 0完成 → phase 1，pending 1
  wait(parity=0)返回 → 安全读本轮TMEM
  round 1 MMA完成通知 → phase 1完成 → phase 0，pending 1
  wait(parity=1)返回 → 安全读本轮TMEM
  round 2 MMA完成通知 → phase 0完成 → phase 1，pending 1
  wait(parity=0)返回 → ……
```

| round | 等待的 parity | 本轮完成后的 parity |
|---|---:|---:|
| 0 | 0 | 1 |
| 1 | 1 | 0 |
| 2 | 0 | 1 |
| 3 | 1 | 0 |

固定传 0 的第二轮有两类时序风险：若在第二轮 MMA 完成前检查，当前 parity 为 1，可能把上一轮完成当成本轮完成，提前读在飞 MMA 的 TMEM；若第二轮已经完成，当前 parity 回到 0，则会等待下一个尚未完成的 phase，而线程卡在等待中不能发出后续工作。

```text
round 1 已完成 → 当前phase 0 / pending 1
              → 错误 wait(0) 等phase 0完成
              → 下一次通知尚未提交，线程不能前进
              → 等待循环不退出（即使外层 rounds=2）
```

外层循环只决定成功结束本轮后是否继续；它不会自动终止内层的 barrier 等待。实测超时并未采集 GPU PC，不能断言每次都停在同一条指令。`tcgen05.wait::ld` 等待 load 完成，也不能替代此前 MMA 的完成等待。

与 3.2 的区别：3.2 在 TMEM 累加四个 K 子块后读一次；本题每轮产生部分结果并读到寄存器，最终在寄存器累加。因此每轮读都必须对应本轮 MMA 的完成。

证据：[修复前](../../experiments/results/m3-mbarrier-20260922T021741Z)、[修复后](../../experiments/results/m3-mbarrier-20260922T030525Z)。




### 3.4 {.prob type=EXPERIMENT file=cuda/m3_tcgen05/04_cta_pair.cu}

比较 `cta_group::1` 和 `cta_group::2` 完成同一个 m256n64k64 任务时的数据开销。

`cta_group::1` 使用两个独立 block；`cta_group::2` 使用一个 cluster 发射一条 M=256 的 mma，两个 CTA 分别保存 B 矩阵的一部分。

先预测，再运行实验：

(a) 在 `cta_group::2` 中，每个 CTA 所需的 B shared memory 是 `cta_group::1` 的多少？TMEM 的占用又如何变化？程序会打印 shared memory 用量，用它验证你的预测。

(b) 使用 Nsight Compute 比较两种实现的 shared memory 总流量。

(c) `cta_group::2` 节省下来的 shared memory 容量，在 M4 的 pipeline 中可以用来做什么？

(d) `cta_group::2` 依赖 sm90 引入的哪一种硬件机制？5090 不支持 2-CTA MMA，结合 0.2 中整理的硬件参数，说明为什么这类机制更常出现在数据中心 GPU 上。

运行：

```
cd assignment02/cuda
make run/m3_tcgen05/04_cta_pair
```

在这个单 tile 实验中，两种实现的运行时间差异处于噪声范围内，因此不要用耗时判断优劣。主要比较 shared memory 用量以及 Nsight Compute 测得的流量。

#### 3.4 解答与实验记录

本题使用给定完整程序，比较同一个 BF16 `M=256,N=64,K=64` 任务。本人最初回答 B 为 1/2、总 shared 为 2/3、TMEM 不变；讨论后总 shared 比例修正为 **5/6**。不能假定 M=N，也不能把 B 减半当成 A+B 减半。

(a) 每 CTA 的 A 为 `128*64*2=16 KiB`，保持不变；B 从 `64*64*2=8 KiB` 减为 4 KiB。所以操作数 shared 为 24→20 KiB，比例 5/6。TMEM 每 CTA 都为 `128*64*4=32 KiB`，两个 CTA 合计 64 KiB，不变。

2026-09-22 B300 实测两种实现均 **PASS，bad=0**。程序打印每 block shared 为 **24588→20492 B**，比纯 A/B 多 12B 管理空间，绝对值减少 4096 B；含管理空间的比值不严格等于 5/6。

(b) NCU 每种实现采集一个 kernel，各有两个 block：

| shared 相关指标 | cta_group::1 | cta_group::2 |
|---|---:|---:|
| SASS shared store bytes | 49162 | 40970 |
| Tensor Core shared wavefront 总量 | 384 | 320 |
| A operand wavefront | 256 | 256 |
| B operand wavefront（对应 1CTA/2CTA scope） | 128 | 64 |
| LSU shared store wavefront | 778 | 652 |

全 grid 的 A/B staging 为 48→40 KiB；store bytes 比纯操作数各多 10B，包含管理写入，不能当成纯 A/B 字节数。两版实测差恰为 8192B。Tensor Core shared 读取的 A 不变、B 减半，总 wavefront 恰好降到 5/6，与预测一致。LSU store wavefront 包含额外写入影响，不强行要求完全按同一比例变化；也不混加不同层级的计数器。

普通运行打印 13.22/14.32 us，仅作原始记录。本题明确要求不以该单 tile 耗时比较优劣；NCU replay 时间更不能作为基准。运行 correctness 超时 5 秒，NCU 超时 35 秒，均额外给 2 秒退出宽限，无自动重试。本次正常完成。

证据：[3.4 实验记录](../../experiments/results/m3-pair-20260922T033128Z/OBSERVATIONS.md)、[原始 NCU](../../experiments/results/m3-pair-20260922T033128Z/ncu-output.txt)、[运行器](../../experiments/modal_m3_pair.py)。

(c) 结合后续 4.3 的流水容量计算：若将本题的 B 复用方式用于 `BM=128,BN=64,BK=64` 的每个 stage，A 保持 16 KiB，B 从 8 降到 4 KiB，每 stage 操作数 shared 从 24 降到 20 KiB。省下的空间可以尝试增加 stage、扩大 tile，或在固定配置下争取更多 block 驻留；每 CTA 的 TMEM 仍为 32 KiB。**这是容量上的机会，不是已实现的 2-CTA pipeline 或已测得的提速。**本段由助教根据 3.4 实测和 4.3 计算起草，待本人复核。

(d) 尚未继续讨论，保留待本人回答。



# 完整 GEMM

本模块将 3.2 的单 tile 实现扩展为完整的 B300 GEMM，并依次加入
tiling、TMA 和多级 pipeline。实验统一使用 4096³ 的 bf16 GEMM，
tile 大小固定为 128×64×64，并与 cuBLAS 结果进行严格相等比较。

::: reading
课件 S084（pipeline 骨架）、S087；Session 4 讲义对应章节；PTX ISA
的 `cp.async.bulk.tensor`、mbarrier（`expect_tx`）小节；CUDA Driver
API 的 `cuTensorMapEncodeTiled`。
:::

下面的表格贯穿 4.1--4.3。第 0 行需要在同一块 GPU 上重新运行
assignment01 Bonus 中的 naive matmul。由于该实现使用 fp32，只比较性能量级。

| 实现 | TFLOPS | 对 cuBLAS 达成率 | 一句话：时间主要花在哪 |
|---|---|---|---|
| naive（assignment01，FP32，BS=16） | 6.3455（B300，4096³） | —（与 BF16 cuBLAS 只比量级） | 每个输出线程沿 K 串行计算；无显式 shared tile 复用或 Tensor Core |
| 4.1 tiled | 39.8（B300，4096³） | 2.26%（同次 cuBLAS 1762.2） | NCU 指向 staging 数据依赖等待，详见 4.1 记录 |
| 4.2 TMA | 519.9（B300，4096³） | 29.46%（同次 cuBLAS 1764.6） | 单缓冲 full/empty 等待，另有输出不合并写回 |
| 4.3 pipeline（S=3） | 469.8（B300，4096³） | 26.62%（同次 cuBLAS 1764.6） | S=3 的 shared memory 限制常驻 block 为 3；TMEM 读回与 barrier 仍有等待 |
| cuBLAS（BF16） | 1762.2–1764.6（各程序同次对照） | 100% | |

上表四级实现在 **同一次 B300 分配**内顺序运行，均通过程序的数值核验，每个程序单独设置 5 秒超时、2 秒退出宽限。naive 保留 Assignment01 原 kernel 与默认 BS=16，只用独立外壳将固定的 1024³ 改成同形状 4096³，并以两次 CUDA event 计时；原始输出、编译命令、源码快照和限制见[同卡梯子记录](../../experiments/results/m4-ladder-20260923T035429Z/OBSERVATIONS.md)。这是一次未锁频运行；4.1–4.3 小节中的早前数值是另次分配的独立记录，不与此表混算。


### 4.1 {.prob type=FROM-SCRATCH file=cuda/m4_gemm/01_tiled.cu}

将 3.2 的单 tile 实现扩展为完整 GEMM：使用 grid 覆盖所有输出 tile，
并沿 K 维循环完成累加。数据 staging 仍然使用 `st.shared` 和 swizzle。

相对 3.2，本题主要新增 tile 映射以及 K 循环中的同步和累加。
具体步骤见文件头。

```
cd assignment02/cuda
make run/m4_gemm/01_tiled
```

通过判测后，填写性能表中 `4.1 tiled` 一行。结合 0.2 中计算的机器
平衡点，判断此时性能主要受哪一环节限制。

#### 4.1 解答与实验记录

本人在学习3.2后重新编写 tiled kernel，助教按请求修正两处 A/B global staging 索引。2026-09-22 在 B300 上编译成功；128×64×64、128×64×128、256×192×256、4096³ 四种形状与 cuBLAS 严格对拍均 PASS（bad=0）。每次5秒超时、2秒退出宽限，无超时。

4096³ 本次普通运行3.38ms、40.7TFLOPS；同次 cuBLAS 1762.4TFLOPS，达成率约2.31%（程序取整打印2%）。这是当前 baseline 实测；后续NCU采集的结论见下文。

主要修正：tile内元素坐标为 iy/ix，A、B 的 global 索引分别为 `(tileM+iy)*K+it*BK+ix`、`(tileN+iy)*K+it*BK+ix`；global 行跨度为 K，不能直接沿用 tile 的 BK 跨度。

证据：[4.1 B300 验证记录](../../experiments/results/m4-tiled-20260922T081102Z/OBSERVATIONS.md)。

NCU显示DRAM利用率0.44%、Tensor pipe活跃度约1.40%、无eligible warp周期82.18%。long scoreboard占平均warp发射间隔89.1%；源码采样主要落在A/B staging中等待global load数据的shared store，支持搬运依赖延迟与重叠不足的解释，不支持HBM已饱和。D按列逐lane写回还产生不合并store，是另一个可独立验证的优化点。具体指标不能直接换算成可实现加速。完整报告已下载：[NCU报告与逐步定位指南](../../experiments/results/m4-tiled-ncu-20260922T081401Z/OBSERVATIONS.md)。




### 4.2 {.prob type=MODIFY file=cuda/m4_gemm/02_tma.cu}

将 4.1 中的 shared memory staging 改为 TMA。

host 端使用 `cuTensorMapEncodeTiled` 创建 tensor map；kernel 中使用
`cp.async.bulk.tensor` 搭配 mbarrier 的 `expect_tx` 完成搬运。
本题仍然使用单缓冲。

tensor map 的参数以及同步方式的变化已经在文件头中给出。
swizzle 由 TMA 根据 tensor map 自动完成，原有 descriptor 不需要修改字段。

```
cd assignment02/cuda
make run/m4_gemm/02_tma
```

通过判测后，填写性能表中 `4.2 TMA` 一行。

比较 4.1 与 4.2，并回答：4.1 中 shared memory staging 的开销由哪些
部分组成？改用 TMA 后，其中哪些工作不再由普通 CUDA 指令完成？

使用 Nsight Compute 辅助分析，观察 4.1 中 SM 时间主要消耗在哪些部分。

#### 4.2 解答与实验记录

在助教提供A tensor map示例、指令讲解后，本人完成B及TMA单缓冲路径。本次review未修改kernel。2026-09-22 B300上四种形状128×64×64、128×64×128、256×192×256、4096³均PASS(bad=0)，沿用4.1判据，非逐位相同比较。每次5秒超时、2秒退出宽限，均正常结束。

同卡重跑4.1为40.6TFLOPS、4.2为523.5TFLOPS，约12.9倍；后者达到同次cuBLAS1762.3TFLOPS的29.7%。

4.1 staging包含线程global load、地址/swizzle计算和shared store，以及这些操作的数据依赖等待。TMA卸载成块搬运和布局转换；NCU执行warp指令由约8.45亿降到2876万，寄存器40→16，Tensor pipe elapsed活跃度1.40%→22.23%。这不是靠增加Tensor Core计算量获得的提升。

4.2仍逐轮TMA→等full→MMA→等empty，不能跨轮重叠。源码采样中两个barrier等待分支占long-scoreboard样本71.1%；输出按列写回的不合并访问仍在。DRAM吞吐仅峰值6.76%，不能称为HBM带宽饱和。前后NCU未锁频且分次采集，加速比采用同卡普通计时，忽略replay期间程序打印的计时。

完整原生报告、源码快照、定位步骤与限制见[4.2 NCU分析](../../experiments/results/m4-tma-ncu-20260922T142749Z/OBSERVATIONS.md)。




::: {.capstone title="prob 4.3(FROM-SCRATCH):多级流水" file=cuda/m4_gemm/03_pipeline.cu}

将 4.2 的单缓冲改为 `STAGES` 级循环缓冲，使后续 K tile 的 TMA
预取能够与当前 tile 的 mma 计算重叠。

`STAGES` 为编译参数，例如：

```
STAGES=4 make -B run/m4_gemm/03_pipeline
```

这里的 `-B` 不能省略，否则修改 `STAGES` 后可能不会重新编译。

文件头给出了每个 stage 使用双 mbarrier 的基本结构，同时包含一个需要
处理的 pipeline hazard：强制发射与机会式预取之间的边界处理错误会导致
死锁。在该错误下，1024³ 可能偶尔通过，而 4096³ 会稳定暴露问题。
开始实现前先阅读文件头说明。

本题不要求 warp specialization、persistent kernel 或 epilogue 融合，
也不设置 cuBLAS 达成率门槛。重点是流水实现是否正确，以及能否根据实验
结果解释性能变化。

```
cd assignment02/cuda
make run/m4_gemm/03_pipeline
cd m4_gemm && ./sweep_stages.sh
```

完成以下内容：

1. 使用 `S=3` 的结果填写性能表中 `4.3 pipeline` 一行。

2. 完成不同 stage 数的性能测试：

   | 形状 | S=2 | S=3 | S=4 | S=6 |
   |---|---|---|---|---|
   | 4096³ | | | | |
   | 256 × 4096 × 16384 | | | | |

   比较两个形状对 `STAGES` 的敏感程度，并结合 shared memory 用量、
   每个 SM 可同时驻留的 block 数以及 block 间并发能够隐藏的延迟进行解释。

3. 任选一个 `STAGES`，画出稳态阶段各 stage 中 TMA 与 mma 的流水时空图。

4. 回答：

   (a) 从 4.1 到 4.3，主要瓶颈发生了哪些变化？

   (b) 梯子表中每一级优化分别减少了哪部分开销？

   (c) 如果继续增大 tile 或增加 stage 数，shared memory 与 TMEM
   哪一个会先成为容量限制？结合 3.4(c) 的结果说明。

:::




#### 4.3 解答与实验记录

本人从 4.2 扩展了每 stage 的 A/B 缓冲和 full/empty barrier，并写了预热、复用与机会式预取；助教按请求修正 `next_to_issue` 的兜底/预取状态流。修改前后源码差异见[本次补丁](../../experiments/results/m4-pipeline-check-20260923T033320Z/opportunistic_change.patch)。

2026-09-23 在 B300 上以 S=3 编译并对拍：128×64×64、128×64×128、256×192×256、4096³ 均 PASS（bad=0），每例 5 秒超时、2 秒退出宽限，无超时。同卡普通运行 4096³：4.2 为 518.7 TFLOPS，4.3 为 470.7 TFLOPS，后者达到 cuBLAS 1754.7 TFLOPS 的 26.8%。这次 S=3 尚未胜过单缓冲，不能单凭“能重叠”推断性能提高。

NCU 原生报告已下载：[4.3 S=3 报告及分析](../../experiments/results/m4-pipeline-ncu-20260923T033628Z/OBSERVATIONS.md)。4.3 的 dynamic shared memory 为 74.75 Kbyte/CTA，受 shared memory 限制最多 3 block/SM；NCU achieved occupancy 18.08%，低于 4.2 旧报告的 41.15%。SASS long-scoreboard 样本中 77.5% 落在 TMEM 读回 `LDTM`，19.7% 落在 barrier 等待分支；输出 store 的不合并访问仍在。NCU 采样分布不等于运行时间分布；两次 NCU 未锁频，性能对照采用同卡普通运行。

2026-09-23 又将同一份源码以 S=2/3/4/6 强制重编，在题目指定的两个形状上各跑一次。八组均通过逐元素对拍（bad=0），每次运行设 5 秒超时及 2 秒退出宽限，均未超时。原始命令、输出与源码哈希见[stage 扫描记录](../../experiments/results/m4-pipeline-sweep-20260923T034401Z/OBSERVATIONS.md)。

| 形状，TFLOPS | S=2 | S=3 | S=4 | S=6 |
|---|---:|---:|---:|---:|
| 4096³ | 512.7 | 476.0 | 397.1 | 239.5 |
| 256 × 4096 × 16384 | 220.6 | 226.7 | 229.6 | 229.3 |

每个 stage 的 A/B 占 24 KiB shared memory；包含 1 KiB 对齐余量后，S=2/3/4/6 分别申请 49/73/97/145 KiB 动态 shared memory。按 S=3 的 NCU 资源配置推算，shared-memory 限制的常驻 block 数依次是 4/3/2/1；其中 S=3 的 3 block/SM 经 NCU 直接确认，其余是资源估算。4096³ 有 2048 个 block，stage 增多降低了 block 间并发，这版实现的预取收益不足以抵消损失。第二形状只有 128 个 block，少于 148 个 SM；增加 stage 不会进一步减少可同时运行的 block 数，S=2 到 S=4 有小幅提升。数据来自单次未锁频运行，细小差异尚不能视为稳定收益。

以 **S=3** 为例，下面画的是 `next_to_issue` 连续推进、各 `empty` 在机会式检查时已经完成的一种合法稳态调度；横向是一轮轮的**发射窗口**，不是按比例绘制的硬件时间。启动时预热先发 `TMA(0/1/2)`；进入稳态后，取任一 `j % 3 == 0` 的轮次，此时 `TMA(j/j+1/j+2)` 已由更早的窗口发出。

```text
时间 →                  窗口 j                  窗口 j+1                    窗口 j+2                    窗口 j+3
Tensor Core             MMA(j) 发射             MMA(j+1) 发射               MMA(j+2) 发射               MMA(j+3) 发射
stage 0 (j % 3)         等 full(j)，消费 j     等 empty(j) → TMA(j+3)     TMA(j+3) 可继续在飞      等 full(j+3)，消费 j+3
stage 1                 TMA(j+1) 已发出         等 full(j+1)，消费 j+1     等 empty(j+1) → TMA(j+4)  TMA(j+4) 可继续在飞
stage 2                 TMA(j+2) 已发出         TMA(j+2) 可继续在飞        等 full(j+2)，消费 j+2     等 empty(j+2) → TMA(j+5)
```

每个 `MMA` 后的 `commit` 才会使相应 `empty` 最终完成；图中的 `TMA(j+3)` 可以与已经发射且尚未完成的 `MMA(j+1)` 重叠。若某次 `try_wait(empty)` 尚未成功，机会式预取停下；到需要消费该 tile 的轮次，代码阻塞等 `empty` 并**保证发射** TMA，然后再等对应的 `full`。所以图说明源码允许的流水关系，不是 NCU 测出的精确重叠时长。

**(a) 瓶颈如何变化。**4.1 用普通线程逐元素做 global load、地址/布局计算和 shared store；NCU 的 long scoreboard 主要落在 staging 的 load→store 数据依赖，Tensor pipe 活跃度仅约 1.40%。4.2 用 TMA 承担块搬运与 swizzle 后，这些普通 warp 指令和等待大幅减少，本次同卡 4096³ 从 39.8 升到 519.9 TFLOPS；单缓冲的 `full/empty` 等待、TMEM 读回及不合并输出 store 成为留下的代价。4.3 允许跨 K 轮预取，但 S=3 的 shared-memory 限制常驻 block 从 4.2 的 8 降到 3，同卡普通运行反而从 519.9 降到 469.8 TFLOPS。NCU 的 long-scoreboard 样本更多落在 `LDTM` 和 `TRYWAIT` 附近，说明原先 staging 的突出问题已转移；样本分布不能直接换算每部分耗时，也不能据此断言 TMA/MMA 实际重叠了多少。

**(b) 梯子每级减少什么。**Assignment01 naive 每个 FP32 输出线程沿 K 串行累加，未显式做 shared tile 复用，也未使用 Tensor Core；4.1 引入 BF16 Tensor Core、输出 tiling 和 shared 中的 A/B tile 复用，但仍支付线程搬运成本。4.2 用 TMA 去掉普通 warp 的逐元素搬运、相关地址/swizzle 指令及其依赖等待。4.3 在 4.2 基础上用额外的 shared 缓冲与 barrier，把后续 TMA 提前到当前异步 MMA 发射之后；它试图隐藏等待，但本次 S=3 的占用损失超过收益。naive 是 FP32，后三级是 BF16，二者的吞吐差距不能全归因于 tiling 或流水，也不应把 naive 除以 BF16 cuBLAS 当成同精度达成率。

**(c) 容量先卡在哪里。**当前每 CTA 的 FP32 累加器占 `128×64×4 = 32 KiB` TMEM（64 列）；增加 `STAGES` 不增加它。A/B 每 stage 却占 `(128+64)×64×2 = 24 KiB` shared memory，S=2/3/4/6 的动态申请为 49/73/97/145 KiB，常驻 block 估算降为 4/3/2/1。因此沿当前参数先增加 stage 时，**shared memory 先限制容量和驻留数**。增大 tile 时两者都可能增长，需按 M/N/K 和合法 MMA 形状重新计算；只作容量外推，在当前 S=3、增大 N 的方向，N=512 时光 A/B 就要 `3×(128+512)×64×2 = 240 KiB`，超过本次 NCU 显示的约 228 KiB/SM shared 配置，而 128×512 的 FP32 累加器恰用满 512 列 TMEM，所以该方向仍是 shared 先到界。这不是声称当前指令支持该单条 MMA 形状。3.4 的 `cta_group::2` 实测将每 CTA 的 B shared 从 8 降为 4 KiB，而每 CTA 的 TMEM 不变；若将这种 B 复用带到流水，每 stage 操作数 shared 可从 24 降到 20 KiB，用省下的空间支持更多 stage、更大 tile 或改善 block 驻留，但不能凭容量节省推断一定更快。

上述时空图和三问由助教根据当前源码及实验记录起草，供本人复核。指定输入的 PASS 不证明所有形状和调度情形均无同步问题；naive FP32 的实测量级已在同卡梯子表列出，但它与 BF16 后三级不构成同精度归因实验。

### 4.4 {.prob type=FROM-SCRATCH opt=Optional}

任选一个方向继续优化：

(a) 实现 4.3 的 `cta_group::2` 版本，利用 B shared memory 用量的减少
增加 pipeline 深度或扩大 tile；

(b) 在 5090 上使用 `mma.sync` 实现相同的 tiling，并与 B300 的结果比较；

(c) 自由优化当前 kernel，提高相对 cuBLAS 的性能，并记录每一步优化
解决了什么问题。

#### 4.4 解答与实验记录

本题按用户要求，由助教在用户的 4.3 `03_pipeline.cu` 基础上逐步实现与实验；原文件没有被这些试验覆盖。所有中间源码、相应上传快照、构建命令、逐次对拍和性能输出见 [4.4 版本索引](../../experiments/m4_44_variants.md)。B300 SXM6 AC，`sm_100f`；各程序 `timeout -k 2s 5s`，无自动重试。选择 `4096³` 作为同次 GPU 分配、两版交替三对的主比较；候选还须通过小形状与长 K 薄形状的严格对拍。**本段实现与书面分析由助教按用户要求完成，待本人复核。**

| 步骤 | 做了什么 | 4096³ 同次分配中位 TFLOPS | 决定 |
|---|---|---:|---|
| [4.3 原版](../../cuda/m4_gemm/03_pipeline.cu) | thread 0 发 TMA/MMA，S=3 | 与 warp 版对照 474.2 | 基线 |
| [warp specialization](../../cuda/m4_gemm/04a_warp_specialized.cu) | warp 0 的 lane 0 连续生产 TMA stage；warp 1 的 lane 0 等 full 后发 MMA，empty 保护复用 | 562.0（+18.5%） | 保留 |
| [persistent 单项](../../cuda/m4_gemm/04b_persistent.cu) | 同一个 CTA 顺次处理多个输出 tile，并跨 tile 复用 TMEM allocation | 517.5（独立对照基线 +9.2%） | 保留思路 |
| [warp + persistent](../../cuda/m4_gemm/04ab_warp_persistent.cu) | 在 warp 分工版上叠加 persistent | 相对 warp 版 654.1（+17.8%）；[直接对照原 4.3](../../experiments/results/m4-44-04ab_warp_persistent-20260923T042936Z/OBSERVATIONS.md) 651.6 vs 472.0（+38.1%） | 当时最佳，见下方追加实验 |
| [单 consumer 2-CTA](../../cuda/m4_gemm/04c_cta_pair.cu) | 每 CTA staged B 减半，组内一次 cooperative MMA 处理 256 行；每 K 轮全员 cluster sync | 280.5 vs 476.2（−41.1%） | 不叠加 |
| [双 consumer 2-CTA](../../cuda/m4_gemm/04d_two_consumer.cu) | 两条 group2 MMA 计算不同的 A 行，复用 staged B；cluster tile M=512、S=2 | 384.2 vs 单 consumer 280.2（+37.1%） | 仍低于单 CTA 基线 |

这里的“双 consumer”不是两个 CTA 把**同一条** MMA 各发一次。PTX 对一条 `cta_group::2` MMA 规定由 pair 中单线程发起；MLC 教程第 9 步是在 CTA 0 用两个 warp 分别发起**两条不同**的 cooperative MMA，各用一份 A、共用 B，输出写入不重叠的 TMEM 列。这一点按用户提供的 [教程 two-CTA 与 multi-consumer 两节](https://mlc.ai/modern-gpu-programming-for-mlsys/zh/chapter_gemm_advanced/index.html#two-cta-cluster) 更正；参考 [PTX ISA issue granularity](https://docs.nvidia.com/cuda/parallel-thread-execution/)。双 consumer 相比单 consumer 的一次比较还同时改变了 M tile 与 stage 深度，不能把 +37.1% 全归给 B 复用。

2-CTA 的每 stage 每 CTA A/B shared 从 24 降至 20 KiB，但本实现每 K 轮全员同步。保存的 [NCU 原生报告](../../experiments/results/m4-44-ncu-20260923T041802Z/OBSERVATIONS.md)显示双 CTA 初版 kernel duration 489.86 µs、SM throughput 19.95%，最佳单 CTA 版分别为 218.59 µs、48.35%；双 CTA 版有 91.05% 的 scheduler cycle 无 eligible warp，CTA barrier 与 memory barrier 等待都突出。两版 achieved occupancy 接近（17.81% vs 17.72%），因此本次减小 B shared 并没有自动转换成速度。NCU replay 下程序打印的时间不是普通 benchmark，吞吐结论以上表的交替计时为准。

尝试利用这份容量将 2-CTA stage 加深到 [S=4](../../cuda/m4_gemm/04c_cta_pair_s4.cu)，最小用例 5 秒超时；进一步试验两种移除逐 K cluster sync 的 [直接 remote TMA barrier](../../cuda/m4_gemm/04e_two_consumer_remote.cu) 与 [本地完成后跨 CTA ready 通知](../../cuda/m4_gemm/04f_two_consumer_handoff.cu)，前者最小用例超时，后者 K=64/128 正确但 K=256 错误。它们均保留源码与失败记录，**不是可用优化**；具体同步错误尚未定位。

Epilogue 融合不做虚构实验：本题输出只有 `D=A×B`，没有要融合的 bias、激活或量化。若仅改变 TMEM 读回和 D store 的布局，那是 epilogue 实现优化，不是算子融合。[CUTLASS 的 GEMM 优化说明](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html)可用于对照 warp 分工、persistent 与 epilogue 的不同目的。上述正确性只覆盖列出的形状；频率未锁定，不能把这组 B300 结果外推到其他尺寸或 GPU。




**4.4 追加实验：epilogue、tile 和 stage（2026-09-23）。** 在 AB 基础上保留每个中间版本，先将逐列 TMEM x1 读回/等待改成 x8，并复用已消费完的 shared stage 作 padded FP32 重排，最后由相邻线程连续写 D；再比较 BN=128/256。没有添加额外数学运算。

| 版本 | 同卡 4096³ 中位 TFLOPS | 相对本轮 AB |
|---|---:|---:|
| AB，128×64×64，S3 | 654.4 | 基准 |
| [G：优化 epilogue](../../cuda/m4_gemm/04g_epilogue.cu) | 970.1 | +48.2% |
| [H128：G 上改 BN128](../../cuda/m4_gemm/04h_bn128.cu) | 1139.9 | +74.2% |
| [H256：G 上改 BN256](../../cuda/m4_gemm/04h_bn256.cu) | 957.7 | +46.3%，低于 H128 |

四版各四个指定形状及三轮 4096³ 全部严格对拍 PASS。[完整数据、源码快照与 NCU](../../experiments/results/m4-epilogue-tiles-20260923T044929Z/OBSERVATIONS.md)。有效 NCU 计数显示 AB→G 的 global-store sectors/request 为 **32→4**，直接支持输出写入合并改善；不是 HBM 流量减少八倍。H128 原 full 报告存在 NaN，已保留并另做关键指标补采。H128 达理论峰值约 **50.7%**、同期 cuBLAS 约 **64.5%**，作为当前 4096³ 最佳版；不是通用最优配置。

随后按本人要求测 S2：保持 BM=BN=128，另一次同卡三轮得到 **BK64 S3=1140.7、BK64 S2=1067.2、BK128 S2=791.3 TFLOPS**，四形状及三次 4096³ 均 PASS。[S2/BK 对照与三份 NCU](../../experiments/results/m4-stage2-20260923T045644Z/OBSERVATIONS.md)确认 shared 限制依次为 **2、3、1 CTA/SM**。BK128 用 3D TMA 分成两个 K64 swizzled slab，未预打包 global 输入；减少控制次数并未带来收益。I/J 编译需明确 `STAGES=2`。

对 stage 的解释：不能把所有 barrier stall 都视为数据搬运不足。本实现有专门的 producer/consumer warp，也有等待两者完成的其他 warp。应分开定位 consumer 等 full 和 producer 等 empty；聚合 stall/occupancy 只能提出假设。S2 在 4096³ 更慢也混有 persistent 网格改变导致的末轮效应，不能全部归因于流水深度。更大 BN 的简化末轮分配反而更差，但速度提高，说明此前不足 50% 并非主要由 tail 决定。所有追加源码、负结果和 diff 见[版本索引](../../experiments/m4_44_variants.md)，书面分析待本人复核。

**补充 tail 对照。** 取 3072×4736×4096，得到 888 个输出 tile，296/444 个 persistent CTA 分别各处理 3/2 个 tile；同卡三轮中位数 **S3=1213.2、S2=1253.0 TFLOPS（S2 +3.3%）**，六次严格对拍 PASS。[原始证据](../../experiments/results/m4-stage-tail-20260923T045944Z/OBSERVATIONS.md)。S2 在本形状获益，说明不能把它在 4096³ 的回退简单归因于流水深度；但形状也变了，仍不能精确分解原形状的 tail 损失。4096³ 保留 H128 S3，S2 保留为形状相关候选。

### 4.5 {.prob type=EXPERIMENT file=cuda/m4_gemm/05_thin_gemm.cu}

观察矩阵形状较窄时 Tensor Core GEMM 的性能变化。

实验形状取自 vLLM 主树中 Kimi K3 的 decode GEMM dispatch 表：

`vllm/models/kimi_k3/nvidia/low_latency_gemm.py`

使用其中七个投影层对应的 N、K，并对 M 取：

$\{1, 8, 16, 64, 256, 1024, 4096, 16384, 65536\}$

其中 N 和 K 由权重形状决定，变化的 M 表示一次 GEMM 处理的 token 数量。
较小的 M 对应 decode batch；较大的 M 可以对应 chunked prefill 中单次
处理的 chunk。vLLM 在 $M \le 16$ 时会放弃 cuBLAS / Tensor Core 路径，
改用 CUDA Core FMA 的 skinny kernel。

运行程序前，先对每个形状计算 arithmetic intensity：

$$
AI = \frac{2MNK}{2MK + 2NK + 2MN}
$$

将结果与 0.2 中得到的机器平衡点比较，并计算对应的理论性能上限：

- compute bound：Tensor Core 峰值；
- memory bound：$AI \times$ 显存带宽。

然后运行：

```
cd assignment02/cuda
make bin/m4_gemm/05_thin_gemm
./bin/m4_gemm/05_thin_gemm <峰值TFLOPS> <带宽GB/s>
```

其中两个参数使用 0.2 中得到的数值。

程序会输出 TFLOPS、GB/s、AI 以及相对于 compute roof 和 memory roof
的达成率。根据结果回答：

(a) 描述性能随 M 变化的趋势。M 较小时，Tensor Core 性能从什么时候
开始明显下降？M 增大到什么范围后进入相对稳定的平台？平台上的达成率是多少？

(b) 对性能下降的形状，比较 compute roof 和 memory roof 两种达成率。
哪些形状主要受到显存带宽限制？

(c) `f_b_proj`（K=128）中两个 roof 的达成率都较低。结合矩阵形状分析，
它的限制来自哪里？

(d) 根据上述结果解释 vLLM 在 $M \le 16$ 时选择 skinny CUDA Core
kernel 的原因。

`in_proj_qkvgfab` 对应 KDA 的输入投影。完成团队题 C1/C2 的同学可以
使用这一行的实验结果作为后续分析的参考。

#### 4.5 解答与实验记录

已于 2026-09-23 在 Modal B300 SXM6 AC 跑完原程序的全部 **63 个形状**，所有命令正常退出，无超时、无重试。使用 0.2 的 **2250 TFLOPS / 8000 GB/s**，机器平衡点 **281.25 FLOP/byte**。实验与书面分析由助教整理，待本人复核。

- [完整 63 点表及实验条件](../../experiments/results/m4-thin-20260923T043531Z/OBSERVATIONS.md)
- [实测前保存的 AI/roof](../../experiments/results/m4-thin-20260923T043531Z/predictions.csv)、[理论与实测合并 CSV](../../experiments/results/m4-thin-20260923T043531Z/measurements.csv)
- [原始 stdout](../../experiments/results/m4-thin-20260923T043531Z/stdout.txt)、[编译、设备、命令及源码 SHA256](../../experiments/results/m4-thin-20260923T043531Z/run.json)、[运行脚本](../../experiments/modal_m4_thin.py)

保留原题程序未改动。CUDA 13.1.80，driver 580.95.05。每个形状先单次调用并同步，warmup 20 次，再用 CUDA event 对 200/50/20 次调用取平均；整轮 timeout 45s，强杀宽限 2s。此处“跑完”指性能扫描完成：原程序没有输出数值对拍，也未检查 cuBLAS API 返回状态。此次未采 NCU，也未实现 skinny 对照。

**实测前的预测。** 所有层 M≤256 都低于机器平衡点；除 f_b_proj 外，其余层从所采样的 M=1024 起进入模型中的 compute-bound 区域。f_b_proj 的 AI 上限为 NK/(N+K)=118.15 FLOP/byte，即使 M 无限大也低于平衡点，对应 memory roof 最高约 945.2 TFLOPS。这里的判断是理想模型预测，不能替代实际瓶颈测量。

**(a) 随 M 的趋势。** M≤16 时，所有层仅达到 0.0%–4.7% 的 TC 峰值（0.0% 是打印舍入，不代表零计算）；相对大 M 已明显退化。M=64 仍普遍很低，M=256–1024 是恢复区。除 K=128 的特例外，多数层在 M≈4096 后进入高吞吐区；并非所有层都严格形成水平平台。

| 投影 | M=16 TFLOPS | M=4096 TFLOPS（TC%） | M=16384 TFLOPS（TC%） | M=65536 TFLOPS（TC%） |
|---|---:|---:|---:|---:|
| f_b_proj | 1.5 | 325.1（14.4%） | 572.7（25.5%） | 692.3（30.8%） |
| q_b_proj | 30.1 | 1748.0（77.7%） | 1913.0（85.0%） | 2039.5（90.6%） |
| o_proj | 93.9 | 1920.7（85.4%） | 2015.7（89.6%） | 2103.3（93.5%） |
| fused_qkv_a_proj | 71.8 | 1753.7（77.9%） | 1838.1（81.7%） | 2052.0（91.2%） |
| in_proj_qkvgfab | 104.9 | 1766.0（78.5%） | 1883.0（83.7%） | 1708.9（76.0%） |
| dense_down_proj | 81.6 | 1846.2（82.1%） | 1855.9（82.5%） | 1641.8（73.0%） |
| dense_gate_up_proj | 99.8 | 1782.1（79.2%） | 1694.0（75.3%） | 1737.5（77.2%） |

因此大 M 高吞吐区约为峰值的 73%–94%；dense_gate_up_proj 在 M=1024 已达 77.4%，而 q_b/o/fused 在更大 M 仍继续改善。in_proj 和 dense_down 在最大 M 有回落。本轮只有一次完整 sweep，不能从这一点归因于某个 kernel 或时钟变化。

**(b) 两种达成率。** 例如 in_proj_qkvgfab 在 M=16 时只有 **4.7% TC peak**，却达 **82.4% memory roof**；M=64 时两者为 **20.7% / 92.8%**。dense_gate_up_proj 在 M=1/8/16 时 memory roof 达成率为 **76.1% / 79.8% / 78.2%**，TC 达成率仅 **0.3% / 2.3% / 4.4%**。这两类大权重小 M 形状最符合权重搬运限制的解释；dense_down 的小 M 为约 62%–65%，也表现出明显供数限制，但未达到带宽上限。

不能把所有“小 M、TC% 低”都判成 HBM 饱和：f_b_proj 小 M 的 memory 达成率仅 1% 左右；q_b_proj 在 M≤256 时也只有约 15%–30%。o_proj 的小 M 虽有 58%–74%，但重复权重可能被缓存，不能仅凭这个值确定 HBM 瓶颈。程序的 GB/s 是逻辑字节量除以时间，**并非 NCU 的实际 HBM 带宽**；上述结论是 roofline 与耗时趋势支持的判断，缓存命中、实际流量和所选 kernel 尚未剖析。

**(c) f_b_proj 的限制。** N=1536、K=128，M 较小时总工作量和可并行 tile 数都有限，K 方向循环又短，难以摊薄启动、调度、数据准备和收尾开销。M=1 到 1024 的耗时基本为 **3.9–4.2 µs**，支持存在固定开销/并行度限制，无法仅用低 AI 解释“双低”。这里 CUDA event 的计时还可能包含连续 host 提交的供给间隙，未测出各部分独立成本。随着 M 增大，短 K 又使输出写回等数据成本占比高，AI 渐近于 118.15，TC 利用率天然受 memory roof 约束。

题干“双 roof 达成率都较低”适用于本次的小 M，不能推广到全部 M：M=65536 时 memory 达成率已为 **73.4%**，而 TC 只有 **30.8%**，这个点主要体现低 AI 的 roof 限制。

**(d) skinny 路径的动机。** 根据题设的 dispatch，M≤16 时使用针对小 M 的 CUDA Core kernel，可以避免通用大 tile 的无效工作，并减少相应数据搬运、同步和流水准备成本；权重读取主导的形状也不需要 Tensor Core 的全部峰值算力。实测显示这些形状的 TC 达成率很低，且有些仅数微秒，支持专门优化小 M 延迟的动机。但本次只跑 cuBLAS，既没有确认其每个形状实际使用哪种指令，也没有对测 skinny kernel，所以不能把题设的 **8%–100%** 提升写作自己的实验收益。




::: lookback

4.1--4.3 从同一个 GEMM 出发，依次加入 tiling、TMA 和多级 pipeline。
回顾梯子表时，重点不是最终的 cuBLAS 达成率，而是能够说明每一步优化
减少了什么开销，以及新的瓶颈出现在哪里。

assignment01 Bonus 中 naive matmul 与 cuBLAS 之间的部分差距，现在已经
可以从 Tensor Core、异步数据搬运和软件流水三个方面解释。进一步的
warp specialization、更大的 tile、epilogue 融合和 persistent kernel
等优化不在本作业要求范围内。

4.5 则说明 Tensor Core 的收益同样依赖矩阵形状。当 M 很小时，即使
kernel 本身使用了前面的优化方法，也不一定能够有效利用 Tensor Core。

:::

# 低精度与 block scaling

降低数值精度可以换取更高的 Tensor Core 吞吐，但代价是可表示的动态范围变小。本模块先从 per-tensor scale 遇到 outlier 时的问题入手，再分析 block scaling 的代数约束，最后完成一条 NVFP4 量化通路，并通过融合实验观察低精度 kernel 的实际性能。

::: reading
课件 S094--S111（fp8 格式：S096；per-tensor 与 outlier：S097--S099；
block scaling 约束：S100--S101；fp4 与 NVFP4：S106--S108；SF 布局：
S108）；`cuda_fp4.h` 中的 `__nv_fp4x2_e2m1`；cuBLASLt 的 block-scaled
FP4 matmul。格式约定见题面材料 `cuda/m5_lowprec/nvfp4_common.h`。
:::


### 5.1 {.prob type=EXPERIMENT file=kernels/quant_outlier.py}

观察 outlier 对 per-tensor scale 的影响。输入包含一万个 $[-1,1]$
均匀分布的元素，并额外加入一个值为 3000 的 outlier。补全脚本中的两个
TODO，完成 per-tensor E4M3 量化与反量化：

```
cd assignment02
uv run python kernels/quant_outlier.py
```

| 采样点 $x\approx$ | 0.5 | 0.1 | 0.01 | 0.005 | 3000 |
|---|---|---|---|---|---|
| 相对误差 | | | | | |

根据实验结果回答：

(a) 去掉 outlier 后重新量化，$x\approx0.5$ 处的误差变化多少倍？

(b) 找出输入被量化为 0 的阈值，并写出该阈值与 scale 的关系式。

(c) 改用 1×128 的 per-block scale 后，包含 outlier 的 block 与不包含
outlier 的 block 分别有什么变化？

#### 5.1 解答与实验记录

尚未完成，保留原题，待后续学习填写。




### 5.2 {.prob type=DERIVE file=kernels/block_scale_sim.py}

block scaling 的代数关键是沿着哪个方向分段 scale 乘积在
K 归约中才能保持常数。补全两个 fp64 模拟函数:

- `gemm_scale_per_row_col`:A 每行一个 scale，B 每个输出列一个
  scale。scale 乘积在整个点积中不变，完整归约后只用乘回一次。
- `gemm_scale_along_k`:A 和 B 的 scale 每 128 个 K 元素改变
  一次。每个 K block 的 partial sum 要分别乘回该段的 scale 乘积，
  然后进行累加。

文件还提供了错误范例 `gemm_scale_along_k_one_restore`：它先把
所有归一化 partial sum 相加，最后只乘回第一段的 scale。

判测:

```
cd assignment02
uv run pytest tests/test_block_scale.py
```

两个正确函数只要在 fp64 容差内与直接 GEMM 等价，不要要求
bit-exact（因为分段会改变浮点加法的分组）。然后回答:

(a) 用两行代数式分别写出 row/column scale 为什么可以在整个
点积外乘回，而 K-block scale 为什么必须逐段乘回。

(b) [NVIDIA CUTLASS 的 Blackwell SM100 GEMM 说明](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html)
把 A 的
scale 布局写成 $M \times \lceil K/SV \rceil$，B 写成
$N \times \lceil K/SV \rceil$，每个 scale 负责连续的 16 或 32 个 K
元素。结合 GEMM 内循环、连续供数和 Tensor Core 指令语义，说明硬件
为什么把 block scale 与 K 归约对齐。

(c) DeepSeek-V3 的 weight 128×128 表示 scale 还在输出通道方向上
共享，但每个组仍覆盖一段 K；NVFP4 则每 16 个 K 元素一组。
结合 5.1(c) 的误差，说明粒度 16 相对粒度 128 有什么优势，又增加了
多少 scale metadata 与供数复杂度。

#### 5.2 解答与实验记录

尚未完成，保留原题，待后续学习填写。



### 5.3 {.prob type=FROM-SCRATCH file=cuda/m5_lowprec/}

完成一条 NVFP4 量化通路。开始前先阅读 `nvfp4_common.h` 中给出的格式约定：

- 每个量化组包含 K 方向连续的 16 个元素；
- SF 使用 `amax / 6`，并以 e4m3 保存；
- SF 按 Tensor Core 消费要求的 swizzled 布局存放，字节偏移公式已经给出。

#### (a) E2M1 编码

在 `e2m1_encode.h` 中实现 host/device 通用的 E2M1
round-to-nearest-even 编码器。

判测会在 GPU 上使用 `cuda_fp4.h` 中的 `__nv_fp4x2_e2m1`
转换同一批候选值，包括所有中点、边界以及采样值，并与自己的编码结果
逐位比较：

```
cd assignment02/cuda
make run/m5_lowprec/03a_encode_check
```

#### (b) NVFP4 quant kernel

在 `nvfp4_quant_kernel.h` 中实现 quant kernel，完成

`bf16 → e2m1 打包 + e4m3 SF + swizzled SF 布局`

设备侧的 E2M1 转换直接使用 `__nv_fp4x2_e2m1`。

第一层判测要求输出逐 byte 与 host 参考结果相等；host 参考使用你在
5.3(a) 中实现的 E2M1 编码器生成。随后将量化结果原样交给 cuBLASLt
的 FP4 matmul，检查生成的数据和 SF 布局能否被实际消费：

```
make run/m5_lowprec/03b_nvfp4_quant
make run/m5_lowprec/test_fp4_gemm
```

正确实现下，FP4 matmul 的 `maxrel` 约为 `4e-3`，主要来自 bf16
输出的舍入误差。如果 SF 布局错位或 scale 对应到了错误的量化组，
通常会出现成块的明显数值错误，而不是仅有小幅舍入误差。

#### (c) Ceiling probe

在 `03c_ceiling_probe.cu` 中实现一个 ceiling probe。它与 quant kernel
使用相同的访存模式，但不执行量化计算，只读取数据并通过简单 xor 后写回。

报告：

- ceiling probe 的 GB/s；
- quant kernel 的 GB/s；
- 两者的比值。

结合 Nsight Compute 判断 quant kernel 距离自己的访存上限还有多远，
以及剩余差距主要来自访存还是计算。

#### (d) Optional

使用 tcgen05 `kind::mxf4nvf4` 直接消费自己生成的 NVFP4 数据，
SF 通过 TMEM 提供，并与 cuBLASLt 的结果对拍。

#### 5.3 解答与实验记录

尚未完成，保留原题，待后续学习填写。




::: lookback

5.3(c) 用来区分量化 kernel 中的访存成本和数值转换成本。作为参考，
软件实现 E2M1 RN-even 时，每个 kernel 约执行 18129 条 FSETP；
使用硬件转换后约为 1272 条 `F2FP.E2M1`，同一 kernel 的时间从
71.7 µs 降到 32.8 µs。Nsight Compute 中的瓶颈也从 SM 利用率
84.7% 的 compute-bound 转为 memory-bound。

这组结果说明，当低精度转换由专用硬件完成后，量化 kernel 可以更接近
纯数据搬运的性能特征；5.3(c) 的 ceiling probe 就用于衡量这一差距。

Hopper 上的 fp8 累加需要软件 promotion（S102--S105，DeepGEMM 使用
双层累加循环），而 Blackwell 进一步在硬件中支持 block scaling
（S105）。本作业不单独设置对应题目，但 5.2 中的 scale 分组约束是
理解两者的共同基础。

:::


::: {.capstone title="prob 5.4(FROM-SCRATCH):融合 rms_norm + NVFP4"}

实现融合的 `rms_norm + NVFP4` kernel：

$$
y = \mathrm{rms\_norm}(x)\cdot w
$$

要求将结果直接量化为 NVFP4，不写回 bf16 中间结果。

本题以 vLLM fused kernel 清单中的 issue #25179，以及两个相关实现
PR #36413、#32957 为背景。相关实现的正确性没有问题，也确实将 kernel
数量从 2 个减少到了 1 个，但当时观察到的端到端收益仍处于 ±1% 左右的
测量噪声内。本题要通过逐形状实验解释这部分收益去了哪里。

按理论字节量计算，两步实现约为 6.56 B/elem，融合后约为
2.56 B/elem，因此单纯从访存量估计可以得到约 2.56× 的理想加速。
这个数字只是上限，实际结果需要通过实验验证。

程序已经提供两步基线、正确性检查和逐形状计时框架。测试覆盖：

- $M$ 从 1 到 16384；
- $K \in \{4096, 7168, 8192\}$；
- 共十个测试形状。

需要完成三部分：

1. 实现融合 kernel，具体线程组织和 tile 结构不限；
2. 分别调优融合实现和两步基线，使两边都使用各自合理的配置后再比较；
3. 完成逐形状性能分析。

第二点必须认真处理。让两步基线处于明显不利的配置下得到的加速比没有
比较意义，这也是相关上游 PR 的实验过程中暴露出的一个问题。

运行：

```
cd assignment02/cuda
make run/m5_lowprec/04_fused_rms_nvfp4
```

报告逐形状结果，包括：

- 实测加速比；
- 融合 kernel 相对 5.3(c) ceiling probe 的性能比例。

重点解释实测结果与 2.56× 理论上限之间的差距：不同 M 范围内主要受到
什么因素限制，并给出 Nsight Compute 数据或计算结果作为依据。

本题不设置性能门槛。实现首先需要保证正确，评分重点放在实验设计和性能归因。

Optional：根据分析得到的主要瓶颈进行一次针对性优化，重新测试并更新表格。

:::




#### 5.4 解答与实验记录

尚未完成，保留原题，待后续学习填写。

### 5.5 {.prob type=CONCEPT}

比较 W4A16 + Marlin kernel（权重 int4、计算 fp16）与 5.3 中的
NVFP4 GEMM。

根据课件 S109 的分类回答：

(a) 两者分别属于存储量化还是计算量化？

(b) 两种方法分别节省哪些资源：显存容量、显存带宽还是计算吞吐？

(c) 在 4.5 中的小 batch decode 场景下，哪一类量化的收益更加直接？

每问用两到三句话回答。

#### 5.5 解答与实验记录

尚未完成，保留原题，待后续学习填写。




# TileLang 对照

前面的模块已经手动处理过 Tensor Core 指令、数据布局和数据搬运。
本模块通过 TileLang 的 lowering 结果观察其中哪些工作可以交给 DSL 完成。

::: reading
课件 S112；assignment01 Module 7（7.5 的表、7.6 的
`kernels/tilelang_matmul.py`）；TileLang 文档中的 lowering/debug 部分。
:::


### 6.1 {.prob type=EXPERIMENT}

使用 assignment01 的 `kernels/tilelang_matmul.py`，或任意使用
`T.gemm` 的 kernel，分别以 `sm_90a` 和 `sm_100a` 为 target 编译。
本题只要求编译，不需要实际使用 H 卡。

保存生成的 CUDA 源码与 lowering 输出，并填写：

| | sm_90a | sm_100a |
|---|---|---|
| 选中的 Tensor Core 指令 | | |
| descriptor 在哪里、由谁生成 | | |
| smem swizzle 布局在哪一步确定 | | |
| 数据由谁搬入 smem | | |

与 M2--M4 中的手写实现进行对照，并回答：

(a) 哪些硬件相关的决策已经由 DSL 自动完成？

(b) 哪些参数仍然需要程序员决定，例如 tile 尺寸和 stages？

最后在 assignment01 7.5 的“谁负责”表中补充一行：

`Tensor Core 指令选择与供数布局`

<!-- 编者注：TileLang 对 sm_100 codegen 的支持范围需在发布前根据
README 中固定的版本重新核实。 -->

#### 6.1 解答与实验记录

尚未完成，保留原题，待后续学习填写。




# 团队选做（推荐） {-}

2--4 人一组，从 `team/` 中的两个题目任选一个，也可以全部完成：

- C1：FlashKDA 官方 kernel 当前使用 SM80 MMA，分析迁移到 SM100 是否值得；
- C2：MiniMax M3 MSA 的小 batch decode 当前使用 Triton，分析是否值得实现专用 kernel。

每道题分为三个阶段：

`复现与测量 → 分析（结论 + 证据）→ 挑战`

挑战阶段不要求一定得到正向加速。如果实验结果表明继续优化收益有限，
只要能够用数据说明为什么不值得继续做，同样视为有效结论。

答辩时间为 10 分钟，另有 5 分钟提问；提问优先由选择另一道团队题的
小组提出。具体要求见 `team/README.md`。

团队题不限制 AI 工具的使用。


# 提交 {-}

- **代码**：提交所有动手题的实现与判测输出；FROM-SCRATCH 题同时保留判测脚本的 PASS 记录。
- **报告**：包含纸面题解答、实验表格与性能归因，以及 DEBUG 题的现象记录和修改说明。所有实验数据注明使用的 GPU。
- **团队题**：提交代码、报告并完成答辩，具体要求见 `team/README.md`。
