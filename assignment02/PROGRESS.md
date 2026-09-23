# Assignment 02：个人学习与实验记录

建立日期：2026-09-11。目标：一起完成 M0–M6 的非团队部分，并积累可核实的简历素材。本文是进度与证据索引，尚未填写的推导、实现和实验不能视为完成。

依据：[正式题面](handout/src/assignment02.md)、各源码文件头、[作业说明](README.md)，以及个人规划 `/Users/simida/Documents/Ohmyresume/docs/coursework-resume/lcpu-infra-2026.md`。本次起点 Git HEAD 为 `0c5c6be`，工作区另有个人修改；每次实验仍须保存实际源码快照或哈希。

## 协作方式与顺序

每题按“读题与原文 → 你推导/实现 → 一起 review → 实际验证 → 你解释结果并写报告”推进。助手可以准备运行与记录工具；作业实现由学员动手。每次实验记录实际操作者，区分给定程序、个人实现和辅助脚本。

用户后续已授权助手使用 simidawhu 的 Modal B300 执行所请求的实验；保留短超时、原始输出和源码快照。课程给定程序、本人实现及助教按请求补全的教学代码分开记录。已做部分的回答与实验统一整理到[handout 个人记录](handout/src/assignment02.md#personal-records)。

主顺序：M0 环境与峰值 → M1 fragment/MMA → M2 descriptor/swizzle → M3 tcgen05 → M4 完整 GEMM 与推理形状 → M5 量化与融合 → M6 TileLang lowering。个人选做单列，主线正确性稳定后继续。团队 C1/C2 不在本次范围。

用户已确认可用 RTX 5090 和 Modal B300：B300 用于主线同机对照，5090 用于 M0/M1 补充验证及 4.4 的可选方向。本机 Mac 用于阅读、编辑及 host 判测。M6 需要 CUDA 编译环境，但不要求实际使用 H 卡。

## 完成清单

状态约定：待开始 / 进行中 / 待实现 / 待实测 / 待解释 / 完成。只有该题要求的代码、验证和报告齐全才标完成。

| 题号 | 要做什么、留下什么 | 环境 | 当前状态 |
|---|---|---|---|
| 0.1 | 运行给定最小 MMA；匹配与不匹配 ARCH 的编译/运行原始现象；PTX/fatbin 证据；自己的解释 | B300；5090 补充 | B300 实测已有证据：100f PASS、120a 运行错误；解释已整理到 handout，5090 尚未运行 |
| 0.2 | 分别推导 5090/B300 的 BF16、FP8、FP4 峰值；声明 dense/sparse、频率和 FMA 口径；官方参数对照、带宽、机器平衡点 | 文档与纸面 | 2026-09-22 按请求补表：dense BF16 209.5/2250TFLOPS，平衡点116.91/281.25；含FP8/FP4路径差异与反算限制，见解答版 |
| 0.3 | 四个判断及理由 | 纸面 | 已回答并 review：见[讨论记录](experiments/m0_concepts.md)，补充 lane 发散与 pipeline/reuse 的区别 |
| 1.1 | 推导 FP8 A/B fragment 的四个坐标函数；host 真值表；解释同一 b32 内元素方向与 load 的关系 | host | 本人第四版 host 判测 PASS：A 的 512 项、B 的 256 项全部匹配；见[最新记录](experiments/results/m1-fragment-20260915T120220Z/results.json)，b32 元素方向与 load 关系已整理到 handout |
| 1.2 | 先保存错误输出和错位规律，再自行修正、解释 fragment 映射错误 | 5090/B300 | 已完成：本人定位 A 行重复并修正；simidawhu/B300 上由 FAIL 59/128 → PASS，固定输入 128 项全部匹配；见[修改与验证记录](experiments/results/m1-bug-20260915T130436Z/OBSERVATIONS.md) |
| 1.3 | 自建 `03_mma_fp8.cu`；手工装载、单 tile e4m3 MMA、f32 累加、CPU 严格对拍；五个 seed 全 PASS | 5090/B300 | 已完成：本人编写 kernel 主体，助教按请求修正 D 映射/打包/头文件复用；2026-09-16 simidawhu/B300 五 seed 全 PASS，见[记录](experiments/results/m1-fp8-20260916T010107Z/OBSERVATIONS.md) |
| 1.4 | 手工 load 与 ldmatrix 两条路径均通过；统计 smem→fragment 的装载/地址指令并解释 | 5090/B300 | 助教按明确请求完成并验证：B300 三 seed 两路径全 PASS；PTX 装载 12→2，地址算术 12→10，打包 12→0；见[实现与讲解记录](experiments/results/m1-ldmatrix-20260920T075433Z/OBSERVATIONS.md) |
| 1.5 | 先预测四种 stride 的 wavefront；NCU 测 wavefront/conflict/cycle，解释比例与耗时差异 | 5090/B300 + NCU | 用户预测 8/16/32/4，B300 NCU 经真实 SASS 指令数归一化后全部吻合；原程序存在重复装载合并，计时限制已记录；见[实验记录](experiments/results/m1-stride-20260920T090419Z/OBSERVATIONS.md) |
| 2.1 | proxy/fence/commit/wait 的排序与理由；三个判断 | 纸面 | 已讨论并整理到 handout：排序及三个判断（错、错、对） |
| 2.2 | descriptor 位域编码、三种布局推导和 host 判测；解释 major 的区别体现在哪里 | host | 部分完成；2026-09-22 当前 host：场景1 LBO FAIL，场景2/3 PASS；本次未改源码 |
| 2.3 | 128/64/32B swizzle；双射和列访问检查；后续由 3.2 硬件验证 | host | 2026-09-22 当前 host 三模式 PASS；128B 有3.2硬件证据；64/32B课程检查与PTX字节映射区别见 handout |
| 3.1 | 五个判断与理由；包含 TMEM 容量计算过程 | 纸面 | 已回答并整理到 handout：对、对、错、对、错，附容量计算 |
| 3.2 | BF16 单 tile tcgen05；五 seed 判测；正确版本后做删除 proxy fence 的观察实验并解释 | B300 | 用户要求助手补全教学实现；2026-09-21 官方五 seed PASS，memcheck/synccheck 各 0 errors；逐段讲解已整理到 handout；去掉 fence 的观察实验待做 |
| 3.3 | 先记录 rounds=1/2/4 的故障；自行修正；画修改前后的 phase/arrival 状态变化 | B300 | 用户改为 round & 1 后 seed42/7 × rounds1/2/4 六例 PASS；修复前超时、phase图及读写关系已整理到 handout |
| 3.4 | 1-CTA/2-CTA：先预测 B smem、TMEM；运行核对；NCU 总流量；容量、流水用途和硬件机制分析 | B300 + NCU | (a/b)已整理；B300两实现PASS，shared 24588→20492B，TC shared wavefront 384→320；(c)助教根据4.3容量分析起草，待本人复核；(d)待回答 |
| 4.1 | tiled GEMM；严格对拍；性能梯子表和瓶颈分析 | B300 | 2026-09-22 B300四形状PASS；4096³ 40.7TFLOPS，cuBLAS1762.4，约2.31%；NCU报告已下载，staging依赖等待及输出不合并写回见解答版4.1 |
| 4.2 | TMA 单缓冲与 tensor map；严格对拍；梯子表、NCU 与 staging 开销分析 | B300 + NCU | 2026-09-22四形状PASS；523.5TFLOPS，同卡4.1的12.9倍、cuBLAS的29.7%；NCU已下载，full/empty等待及输出不合并见解答版4.2 |
| 4.3 | 多级流水；S=3 梯子表；两个形状各扫 S=2/3/4/6；稳态流水时空图；容量/驻留/瓶颈解释 | B300 + NCU | 2026-09-23 S=3 四形状及 stage 扫描八组全部 PASS；NCU 报告已下载；同一次 B300 分配重跑含 A01 naive 的 4096³ 梯子，表、时空图和三问均已写入解答版；书面分析由助教起草，待本人复核 |
| 4.4（选做） | 从 2-CTA pipeline、5090 MMA 路线、自由优化选一个方向；保留每一步的动机与证据 | B300 + NCU | 追加 epilogue/tile/stage 实验已完成：4096³ 同卡 AB 654.4→G 970.1→H128 1139.9 TFLOPS（+74.2%）；S2 BK64/BK128 在该形状回退，全部指定对拍 PASS；888-tile 对照中 S2 比 S3 快 3.3%，保留为形状候选。NCU 原件、异常补采、源码 diff 均已保存，[版本索引](experiments/m4_44_variants.md)与解答版待本人复核 |
| 4.5 | 七投影 × 九个 M；实测前计算 AI/roof；填 63 点数据并回答四问 | B300 | 2026-09-23 原程序 63 点扫描完成，无超时；预计算和实测 CSV 已保存，四问已写入解答版，待本人复核；性能实验无数值对拍/skinny 对照 |
| 5.1 | 两个 TODO；五点量化误差表；无 outlier、归零阈值、1×128 block scale 对照和解释 | host Python | 待实现 |
| 5.2 | 两个 FP64 scale 模拟函数；三项 pytest；代数、K 方向供数和粒度取舍解释 | host Python | 待实现 |
| 5.3(a) | E2M1 RN-even 编码，与硬件转换逐位核对 | B300 | 待实现 |
| 5.3(b) | NVFP4 quant、packing、SF 布局；逐 byte 核对及 cuBLASLt 消费验证 | B300 | 待实现 |
| 5.3(c) | 同访存模式 ceiling probe；quant/probe GB/s 与比例；NCU 归因 | B300 + NCU | 待实现 |
| 5.3(d)（选做） | tcgen05 直接消费 FP4，与 cuBLASLt 对拍 | B300 | 主线后推进 |
| 5.4 | RMSNorm+NVFP4 融合；两步与融合分别合理调优；10 形状正确性、延迟、ceiling 比例及瓶颈证据 | B300 | 待实现 |
| 5.4 追加优化（选做） | 根据已测瓶颈再做一次针对性优化，更新实验表 | B300 | 主线后推进 |
| 5.5 | W4A16+Marlin 与 NVFP4：类别、节省资源、小 batch 收益；每问 2–3 句 | 纸面 | 待开始 |
| 6.1 | 固定 TileLang 0.1.13，分别编译 sm_90a/sm_100a；保存 CUDA/lowering、填四行对照表、答职责两问；补 A01 7.5 一行 | CUDA 编译环境 | 待开始 |

## 判测与实验入口

以下路径相对 `assignment02/cuda`；运行前先完成对应实现。B300 默认 `ARCH=100f`，5090 使用 `ARCH=120a`。改变 ARCH、STAGES 或相关头文件后用 `make -B`，以免复用旧二进制。

| 范围 | 入口 |
|---|---|
| M0/M1 的已有文件 | `make -B run/m0_env/01_first_mma`、`make -B run/m1_sm80/<文件名去掉.cu>` |
| 1.3 | `make -B bin/m1_sm80/03_mma_fp8`；在 `m1_sm80` 运行 `judge_mma_fp8.sh` |
| 2.2/2.3 | `make -B run/m2_smem/02_descriptor` / `03_swizzle` |
| 3.2 | `make -B bin/m3_tcgen05/02_single_tile` 后在 `m3_tcgen05` 运行 `timeout -k 5 180 ./judge_tile.sh` |
| 3.3 | `make -B bin/m3_tcgen05/03_bug_mbarrier` 后在 `m3_tcgen05` 运行 `timeout -k 5 210 ./judge_mbar.sh` |
| 4.1–4.3 | 对应 `make -B bin/m4_gemm/...`；`judge_ladder.sh` 汇总检查；`sweep_stages.sh` 扫描 |
| 4.5 | `make -B bin/m4_gemm/05_thin_gemm` 后运行程序，传入自己在 0.2 推导的峰值 TFLOPS、带宽 GB/s |
| 5.1/5.2 | 回到 `assignment02`：`uv run python kernels/quant_outlier.py`；`uv run pytest tests/test_block_scale.py` |
| 5.3 | `make -B run/m5_lowprec/03a_encode_check`、`03b_nvfp4_quant`、`test_fp4_gemm`、`03c_ceiling_probe` |
| 5.4 | `make -B run/m5_lowprec/04_fused_rms_nvfp4` |
| 6.1 | `uv sync --extra tilelang`；在自己写的编译导出入口中复用 A01 `kernels/tilelang_matmul.py` |

本机的纯 host 例子（这是调用原有判测，不修改题目）：

```bash
cd /Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/cuda
clang++ -x c++ -std=c++17 -O2 m1_sm80/01_fragment_map.cu -o /private/tmp/a02_fragment_map
/private/tmp/a02_fragment_map
```


## 实验记录格式

每次在 `experiments/results/<实验名-UTC时间>/` 留原始输出及元数据，报告引用该目录。记录：题号、操作者、时间、GPU/驱动/CUDA、目标架构、编译命令、源码快照/哈希、形状/dtype/seed、参数、退出码、正确性和独立计时结果。NCU 与普通计时分开；不能用 profiler 重放时间当基准。

| 实验 ID | 必须保留的对照与条件 | 原始证据 | 本人解释/报告 |
|---|---|---|---|
| L02-ENV | 0.1 匹配/不匹配 ARCH、编译和运行阶段、生成 PTX 与实际 binary 的区别 | [B300 实验记录](experiments/results/m0-20260913T065647046705Z/OBSERVATIONS.md) | 待本人解释 |
| L02-LAYOUT | host 映射与 GPU 对拍分别记录；MMA shape、dtype、stride；PTX/SASS 指令及 NCU wavefront | 见 handout 个人记录中的 M1–M3 证据链接 | 已整理已做部分；未完成项单列 |
| L02-GEMM | 4096³ BF16、128×64×64 tile；tiled→TMA→pipeline(S=3)→cuBLAS；同卡重跑 A01 naive FP32 仅对比量级 | [同次 B300 梯子](experiments/results/m4-ladder-20260923T035429Z/OBSERVATIONS.md)：四级均 PASS，含源码快照及命令 | [解答版梯子表与 4.3 三问](handout/src/assignment02-with-answers.md#完整-gemm)，助教起草待本人复核 |
| L02-STAGES | 4096³ 与 256×4096×16384；各 S=2/3/4/6；smem/CTA、驻留数与流水图 | [B300 stage 扫描](experiments/results/m4-pipeline-sweep-20260923T034401Z/OBSERVATIONS.md)：八组 PASS、时延及资源估算 | [解答版 4.3](handout/src/assignment02-with-answers.md#43-解答与实验记录)已填扫描表、敏感度解释和 S=3 时空图，助教起草待本人复核 |
| L02-THIN | 七投影 × M=1/8/16/64/256/1024/4096/16384/65536；理论 roof 与实测分列 | [B300 63 点](experiments/results/m4-thin-20260923T043531Z/OBSERVATIONS.md)，含预测、实测 CSV、源码快照及原始日志 | [解答版 4.5](handout/src/assignment02-with-answers.md#45-解答与实验记录)，助教整理待本人复核 |
| L02-QUANT | 编码硬件核对、quant bytes、cuBLASLt 消费；量化粒度、SF 布局、同访存 probe | 待填 | 待写 |
| L02-FUSION | 题目 10 形状；两步/融合各自 launch 参数；数据及 SF 的验证范围；延迟与 ceiling 比例 | 待填 | 待写 |
| L02-LOWERING | TileLang/compiler 版本、两个 target、相同 kernel/参数、生成 CUDA/lowering | 待填 | 待写 |

求职加做单列在必做证据之后：例如随机 BF16 容差验证、更多形状、端到端引擎集成。给定 baseline、理论上限、上游性能数字与个人优化收益分别记录；只有实际做过并验证的内容才回填简历。

## 起点核查与实测

2026-09-11，助手在本机执行未修改的纯 host 骨架：

- 平台：Darwin arm64；Apple clang 17.0.0；当前 PATH 未发现 `nvcc`/`nvidia-smi`。
- 三个程序均编译成功。1.1 报 766 mismatches；2.2 三场景 FAIL；2.3 三模式均 FAIL。
- 这些是尚未实现的骨架诊断，不是已完成的作业，也不是 CUDA/GPU 性能证据。
- 原始记录：[host-scaffold-20260911T065641Z/results.json](experiments/results/host-scaffold-20260911T065641Z/results.json)。
- `assignment02/.venv` 尚未创建，现有 A01 venv 未安装 torch。5.1/5.2 尚未运行。
- 已在临时隔离环境安装 Modal CLI 1.5.5，未改动课程 Python 依赖。2026-09-13 已成功连接并执行 B300 实验。
- 新建辅助运行器 [experiments/modal_m0.py](experiments/modal_m0.py)，语法检查通过。CPU 镜像构建匹配/不匹配 ARCH 两个 binary；GPU 函数只运行一次，B300 单卡、120 秒超时、无重试；保存源码、哈希、编译输出、PTX/fatbin 列表和运行结果。
- 云端运行在执行前被自动审批拒绝，原因是三个课程文件向 Modal 的具体上传授权尚不明确。本地已确认它们与 HEAD 一致、无个人修改；公开上游内容的在线复核未成功，因此没有重试上传，也没有生成本次 B300 实测结果。
- 2026-09-13：用户明确同意上述三个课程文件及辅助运行器上传至其 Modal 工作区并执行一次 B300 检查（单卡、GPU 函数上限 120 秒、无重试）。该授权继续有效；用户同时要求先讲解 PTX `wmma.mma`，特别是 register 排布，因此本轮先讲解，尚未启动云端实验。
- 2026-09-13，用户要求开始实验后执行 0.1。首轮因辅助脚本读取 Modal 版本元数据失败，命令结果未返回；保留错误并修正记录逻辑后重新采集，第二轮完整成功。两轮自动重试均为 0。B300 100f 版 PASS；120a 版编译成功、运行返回 `cudaErrorNoKernelImageForDevice`。原始输出、PTX、编译参数、真实输入快照及错误历史见 [实验记录](experiments/results/m0-20260913T065647046705Z/OBSERVATIONS.md)。

讲解后继续实验时，从仓库根目录使用当前临时环境运行（临时依赖缓存如已失效需重建）：

```bash
uv --cache-dir /private/tmp/a02-uv-cache run --offline --no-project --with modal==1.5.5 python -m modal run assignment02/experiments/modal_m0.py
```

Modal 镜像显式添加的课程文件是 `cuda/common.h`、`cuda/Makefile`、`cuda/m0_env/01_first_mma.cu`；Modal 还会发送辅助运行器的执行代码。运行记录写入 `experiments/results/m0-<UTC时间>/`。辅助运行器不包含任何题目 TODO 的实现。

## 当前学习单元：0.1

先读 `cuda/m0_env/01_first_mma.cu` 的文件头、kernel launch 与 inline PTX；配合课件 S008–S021 和 [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/) 的 mma 章节。先理解整体数据流，fragment 公式留到 M1 细推。

你先尝试说明以下三点，我们再结合实际输出 review：

1. 这条指令处理的矩阵尺寸分别是什么，输入与累加精度是什么？
2. 为什么程序启动的是一个 warp；这个 warp 共同产生多大的输出？
3. 对不匹配的 ARCH，你预测在哪一步出现问题：编译、加载还是 kernel launch？用实验验证预测。

记录模板：

- 匹配 ARCH 的命令、退出码与输出：100f 编译/运行均退出 0，输出 PASS，已保存。
- 不匹配 ARCH 的命令、退出码与原始错误：120a 编译退出 0、运行退出 1；在 kernel launch 后的错误检查处报告 `cudaErrorNoKernelImageForDevice`，已保存。
- PTX/fatbin 中看到的 target 与代码内容：独立 PTX 为 `.target sm_100f`，含 m16n8k16 MMA；两 binary 的 ELF/PTX 列表已保存，详见实验记录。
- 自己关于 fatbin/JIT 的解释：待写。

## 提交前回看

代码、所有动手题判测输出、FROM-SCRATCH 的 PASS 记录，以及包含纸面题、实验表、归因、DEBUG 修改前后说明的报告都要齐全。数据注明 GPU；只有编译、只有 host 检查、尚未运行的部分如实标注。

已发现的操作细节：`judge_tile.sh` 没有自身 timeout；stage sweep 会在个别失败后继续，必须逐项检查；扫完 S=6 后回填 S=3 要重新编译。5.4 给定检查侧重数据 bytes，PASS 不自动证明 SF bytes 已被验证；完整实现 review 时再补足相关证据。
