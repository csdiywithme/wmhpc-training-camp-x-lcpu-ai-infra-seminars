# 4.3 S=3 流水：B300 正确性与 NCU

2026-09-23，NVIDIA B300 SXM6 AC，CUDA 13.1.80，NCU 2025.4.1。按 `sm_100f`、`-lineinfo` 编译 4.2 和 4.3；上传源码与下载报告的 SHA256 均核对通过。四种形状 128×64×64、128×64×128、256×192×256、4096³ 均按程序的逐元素数值比较通过（`bad=0`）。普通运行每例 `timeout -k 2s 5s`，NCU 限时 35 秒，无超时或自动重试。

## 同卡普通运行

| 实现 | 4096³ 时间（程序四舍五入） | TFLOPS | 同次 cuBLAS TFLOPS |
|---|---:|---:|---:|
| 4.2 TMA 单缓冲 | 0.26 ms | 518.7 | 1753.4 |
| 4.3 pipeline S=3 | 0.29 ms | 470.7 | 1754.7 |

S=3 为 cuBLAS 的约 26.8%，比本次 4.2 吞吐低约 9.3%。这是同一容器、同一 GPU 的单次普通运行对照，尚未完成多次重复或 S=2/4/6 扫描；不要把这一个结果当最佳 stage 数。

## NCU 指向的代价

| 指标 | 4.2 旧 NCU 报告 | 4.3 S=3 本报告 |
|---|---:|---:|
| Dynamic shared memory / block（NCU Kbyte） | 25.60 | 74.75 |
| Shared-memory 限制下最多常驻 block / SM | 8 | 3 |
| Achieved occupancy | 41.15% | 18.08% |
| Tensor pipe elapsed 活跃度 | 22.23% | 19.90% |
| 执行指令 | 28,760,345 | 41,011,189 |
| 无 eligible warp 周期 | 89.24% | 86.35% |

本次配置每 block 使用 3 份 A/B shared buffer，显著降低可驻留 block 数。预取减少了部分等待，但更多发射与 barrier 管理指令、较低的 block 并发共同限制收益。NCU 两列来自不同分配且未锁定 GPU 频率；具体性能差异以上面的同卡普通运行为准。Tensor pipe 活跃度不是理论 FLOPS 达成率，无 eligible warp 比例也不能单独决定吞吐。

SASS 源码采样的 long-scoreboard 样本共 5,261 个，其中 `LDTM` 上 4,078 个（77.5%），与 `TRYWAIT` 相邻的等待分支上 1,034 个（19.7%）。相比 4.2，采样热点从 barrier 等待分支更多地落到了 TMEM 读回所在的 epilogue；采样比例不是总运行时间比例。首个等待热点 PC `0x2af5515ad830`，前一条是 `SYNCS.PHASECHK.TRANS64.TRYWAIT`。输出写回的多余 global sector 仍为 14,680,064，与 4.2 相同，原有的不合并 store 尚未改变。

NCU 报告的 kernel duration 为 294.59 µs。**NCU replay 时程序打印的 414.67 ms 不能用作普通性能**。报告使用 `--clock-control none`，六项 CTC 互连指标不可访问；本分析未使用这些指标或把 NCU 的规则预估收益当成实测加速。

## 查看报告

用 Nsight Compute GUI 打开 [`pipeline_4096.ncu-rep`](pipeline_4096.ncu-rep)。先看 Occupancy 的 `Block Limit Shared Mem`，再在 Source → SASS 按 long-scoreboard 采样排序，检查 `TRYWAIT` 和 `LDTM`；输出写回可看 global excessive sectors。原始导出为 [`ncu-details.txt`](ncu-details.txt)、[`ncu-source.txt`](ncu-source.txt)、[`ncu-raw.csv`](ncu-raw.csv)，命令与校验记录为 [`run.json`](run.json)，测试源码在 `inputs/`。

本报告采集时，S=2/4/6、第二形状、时空图和书面归因尚未完成。后续已另存[stage 扫描](../m4-pipeline-sweep-20260923T034401Z/OBSERVATIONS.md)和[同卡 4096³ 梯子](../m4-ladder-20260923T035429Z/OBSERVATIONS.md)，时空图与三问见[解答版 handout](../../../handout/src/assignment02-with-answers.md#43-解答与实验记录)。
