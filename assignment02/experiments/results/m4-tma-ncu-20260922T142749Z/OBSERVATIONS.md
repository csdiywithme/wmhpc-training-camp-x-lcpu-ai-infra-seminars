# 4.2 TMA：B300 验证与 NCU 分析

2026-09-22，NVIDIA B300 SXM6 AC，CUDA 13.1，NCU 2025.4.1。保留输入源码快照、PTX、命令输出和原生报告；未修改学员 kernel。报告下载后 SHA256 与远端记录一致（7,575,055 字节）。

## 正确性与普通计时

128×64×64、128×64×128、256×192×256、4096³ 均 PASS(bad=0)。沿用程序与 cuBLAS 的误差判据，并不表示逐位相同。每条运行限时5秒，退出宽限2秒；NCU限时35秒，无超时。

同一 Modal 容器、同一张 GPU 普通运行：

| 实现 | 4096³ 时间 | TFLOPS | 同次 cuBLAS TFLOPS |
|---|---:|---:|---:|
| 4.1 tiled | 3.38 ms | 40.6 | 1764.8 |
| 4.2 TMA | 0.26 ms | 523.5 | 1762.3 |

TMA 约快12.9倍，达到cuBLAS的29.7%。时间仅显示两位小数，倍率按吞吐计算。NCU replay期间程序打印的272.67ms不能当普通性能。

## 为什么变快

4.1 由线程执行 global load、地址和swizzle计算、shared store，必须等输入寄存器就绪才能继续。4.2 用TMA完成成块搬运与布局转换，显著减少线程指令及寄存器压力。

| NCU指标 | 4.1旧报告 | 4.2本次报告 |
|---|---:|---:|
| 执行warp指令 | 844,697,600 | 28,760,345 |
| 寄存器/线程 | 40 | 16 |
| Tensor pipe elapsed活跃度 | 1.40% | 22.23% |
| DRAM吞吐占峰值 | 0.44% | 6.76% |
| 无eligible warp周期 | 82.18% | 89.24% |

前后NCU不是同次分配，且未锁频；精确加速比使用上面的同卡普通计时。无eligible比例没有下降不代表变慢：卸载搬运后线程指令更少，执行和等待的构成也变了。Tensor pipe活跃度不是理论FLOPS达成率。

## 剩余瓶颈与源码定位

当前单缓冲顺序为 `TMA → wait(full) → MMA → wait(empty) → 下一轮TMA`，没有在本轮计算时提前搬下一轮。NCU long scoreboard占平均warp发射间隔64.3%，不是总运行时间的64.3%。

SASS PC采样共有10,686个long-scoreboard样本：

- `0x2abfe95ac9a0`：5,849个，前一条为 `SYNCS.PHASECHK.TRANS64.TRYWAIT`，对应等full（源码163行）。
- `0x2abfe95ace30`：1,744个，前一条同为TRYWAIT，对应等MMA完成（源码196行）。

两处合计71.1%的long-scoreboard样本。采样落在依赖barrier谓词的分支上；不能仅凭long scoreboard名称推断仍是global load。源码串行依赖与采样共同支持优先研究下一题pipeline；不保证增加任意stage都提速。

另一个独立问题是输出写回（206行）：固定n时相邻lane写不同行，地址相差N*4字节。每warp的128字节有效数据占32个32字节sector，理想连续写只需4个。64条STG合计多余sector 14,680,064，与4.1一样。它占本次global sector的87.5%（旧版12.5%）；绝对浪费没变，比例增大源于普通输入load被TMA替代，不能解释为写回恶化了7倍。

当前证据支持等待/重叠不足以及输出不合并，不支持HBM带宽已饱和。不要将NCU规则的预估收益相加或当作实测加速。

## 在GUI中查看

打开 [tma_4096.ncu-rep](tma_4096.ncu-rep)，选择gemm_tma：

1. Summary查看Duration（262.75us）、Memory Workload及Scheduler Statistics。
2. Source切到SASS，按long scoreboard采样降序，找上述两个PC，连同前面的TRYWAIT阅读。
3. 查看STG的global excessive sectors，联系源码206行的lane地址跨度。
4. 对照[旧报告分析](../m4-tiled-ncu-20260922T081401Z/OBSERVATIONS.md)，区分已经消除的staging依赖和仍存在的epilogue。

报告嵌入源码，完整原始输出见 [run.json](run.json)、[ncu-details.txt](ncu-details.txt)、[ncu-source.txt](ncu-source.txt)、[ncu-raw.csv](ncu-raw.csv)。六项CTC互连指标不可访问；本分析未使用它们。仅验证上述四个合法整tile形状，未声称支持任意尺寸或做过完整sanitizer检查。
