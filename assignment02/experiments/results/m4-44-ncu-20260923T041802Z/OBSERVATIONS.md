# 4.4 B300 Nsight Compute 报告

同次 Modal B300 SXM6 AC 分配，CUDA 13.1、NCU 2025.4.1；两版均 `sm_100f -lineinfo` 编译。`--set full --clock-control none`，单 kernel 采样，单程序外部超时 35 秒，无自动重试。两个原生 `.ncu-rep` 已下载到本目录，SHA256 与源码快照见 `run.json`；可直接用 Nsight Compute GUI 打开。

| 指标 | warp+persistent | 单 consumer 2-CTA |
|---|---:|---:|
| Kernel duration | 218.59 µs | 489.86 µs |
| Compute (SM) throughput | 48.35% | 19.95% |
| Scheduler `No Eligible` | 88.85% | 91.05% |
| Achieved occupancy | 17.72% | 17.81% |
| Block limit: shared memory | 3 | 3 |

2-CTA 版的 SM 活跃度低、无可发射 warp 比例高；Warp State 还报告平均每次发射间隔中约 11.6 cycle 等 CTA barrier、10.8 cycle 等 memory barrier，符合源码每轮全员 `cluster.sync()` 和本地 full/empty 等待的高同步成本。这里的 NCU 占用比例接近，**不能**把 2-CTA 回退简单归因于 shared 容量。NCU 的建议百分比是分析线索，不是可相加的预测加速比；普通吞吐结论以同次分配交替计时为准。NCU replay 下程序自身打印的耗时不作 benchmark。

[最佳版原生报告](04ab_warp_persistent_4096.ncu-rep) · [2-CTA 原生报告](04c_cta_pair_4096.ncu-rep) · [详情导出](04ab_warp_persistent-ncu-details.txt) · [2-CTA 详情导出](04c_cta_pair-ncu-details.txt)
