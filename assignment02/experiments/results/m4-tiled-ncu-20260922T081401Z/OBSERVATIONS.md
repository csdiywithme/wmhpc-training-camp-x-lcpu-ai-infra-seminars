# 4.1 Nsight Compute 瓶颈定位

2026-09-22，B300 SXM6 AC，148 SM，CUDA13.1.80，driver580.95.05，NCU2025.4.1.0。源码未改，仅编译增加 -lineinfo。对同一二进制先普通运行，对拍PASS：4096³，3.37ms，40.8TFLOPS；cuBLAS1765.1TFLOPS。此前不带lineinfo的运行40.7TFLOPS。

## 本地报告与采集口径

- [GUI报告](tiled_4096.ncu-rep)：已下载，SHA256与run.json一致，约81MiB，包含源码；使用2025.4.1或更高且兼容的Nsight Compute打开。
- [Details文本](ncu-details.txt)、[Raw指标CSV](ncu-raw.csv)、[SASS与源码计数器文本](ncu-source.txt)、[热点摘要](source-hotspots.json)、[完整元数据](run.json)。
- 编译 -O2 -lineinfo；--set full --kernel-name regex:gemm_tiled --launch-skip 21 --launch-count 1：跳过1次对拍launch及20次warmup，采集第一个计时区间的目标kernel。其他cuBLAS kernel不匹配。
- --import-source yes --source-folders /opt/m4_tiled/cuda；报告包含源文件，云端路径不在本机也可查看导入源码。
- timeout -k 2s 35s，40 passes正常结束，无GPU重试；--clock-control none，时钟未锁定，默认replay缓存控制会影响缓存状态。
- 报告kernel Duration=4.08ms；不要用被profiling干扰的程序整体打印565.78ms或其中cuBLAS结果做性能对照。普通计时才是baseline。
- 六个CTC互连指标不可访问，其余报告已成功导出；本题不是互连分析。所有三种离线export退出码0。

## 按顺序读GUI

1. 打开report，选gemm_tiled，先到Details → GPU Speed Of Light。Compute(SM)=17.68%，DRAM=0.44%，L2=4.01%，Tensor pipe active=1.404448%（raw中sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed）。Compute(SM)不是Tensor Core FLOPS达成率。首先排除“HBM带宽已经跑满”的解释。
2. Details → Scheduler Statistics：平均7.05 active warps/scheduler，只有0.21 eligible warps/scheduler，82.18%的周期没有eligible warp。已有驻留warp，但多数不能发射。
3. Warp State Statistics：每条issued instruction平均间隔39.55 warp cycles，其中long scoreboard约35.2 cycles，占89.1%。这是warp等待统计，不是说89.1%的GPU墙钟时间都可消除。该类别指向L1TEX相关数据依赖，下一步必须定位生产这些数据的指令。
4. Source → CUDA-C/SASS，选Warp Stall Sampling，按stall_long_sb排序。热点在A/B staging赋值（快照01_tiled.cu第107/112行）对应的ST.E.U16。八条shared目的store共有255523个long-scoreboard样本，所有SASS指令共259962个，约98.3%落在这些store上。举例：LDG.E.U16 R31后面的ST.E.U16 ... R31在等待同一个源寄存器。停在store不表示shared带宽满了，它首先要等global load提供数据。报告地址空间也把这些store归为Shared。
5. 回到Memory Workload Analysis：L2 hit=87.16%，无local/shared spilling。结合低DRAM利用率，证据指向标量global→register→shared路径的数据依赖延迟和不足的重叠，不支持直接说“显存带宽瓶颈”。仅凭这份报告不能量化具体是哪一级cache延迟贡献多少。
6. Occupancy：理论50%，实际44.19%；40 registers/thread，shared限制8 blocks/SM。提高occupancy不是看到低性能后的唯一答案，首先看为什么驻留warp不可发射。

## 另一个已定位的问题：输出写回

Source的L2 Theoretical Sectors Global Excessive落在epilogue STG上。固定n时相邻lane写不同的行，相隔N*4字节，因此写回不合并。本次多出14680064个sector：64条store各229376个；总global sectors的约12.5%。这不是A/B输入未合并。它值得单独优化，但不能用这个比例直接当作总时间占比，更不能把NCU估计加速率当成实测收益。

## 下一步如何做对照

先沿课程4.2保持tile、输入和计时不变，只把staging改成TMA；对拍通过后比较普通耗时、long scoreboard热点、eligible warps和Tensor pipe活跃度。这样检验搬运路径是否是主要改进方向。再在4.3加入pipeline，检验搬运与计算重叠的收益。epilogue合并写回可另做独立对照，避免一次改多处后无法归因。本次没有修改kernel进行优化。

官方阅读：[Nsight Compute GUI](https://docs.nvidia.com/nsight-compute/NsightCompute/)、[指标与stall解释](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)。
