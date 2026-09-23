# NCU 独立审查：20260918T092500Z

**六组硬件计数器采集全部成功；原 `ncu_summary.json` 的 `failed` 是解析器误报。** 每个 profile 的采集、导出退出码均为 0，日志记录 9 次 replay，`.ncu-rep` 有效，CSV 均为 329 列、3 行：指标名、单位、一个 kernel 的数据。旧解析器期待 long CSV 中的 `Metric Name` / `Metric Value` 列，未识别本次 `--page raw` 的 wide CSV。原始 summary、CSV、报告和日志均未修改；恢复后的原始值、单位、来源 hash 与衍生值保存为 [ncu_metrics.json](ncu_metrics.json)。

采集采用 `--replay-mode kernel --cache-control all --clock-control none --launch-count 1`。以下耗时是 NCU profile 的单个 kernel 观测，用于解释机制；正式性能排名应使用重复、随机顺序 benchmark。9 次 replay 分属计数器采集轮次，不是 9 次独立计时统计。动态频率及 replay 会影响跨次比较。

## 计算活动与任务尾部

`Tensor/elapsed` 使用整个测量区间为分母，`Tensor/active` 使用 SM 活跃周期为分母。它们是管线周期指标，不是按 FP16 算术操作数计算的标称 2250 TFLOPS 百分比。[NVIDIA 指标定义](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-structure)

| case / variant | 逻辑任务 / 轮数 | profile μs | 按 2MNK 计算 TFLOPS | SM active/elapsed | Tensor/elapsed | Tensor/active |
|---|---:|---:|---:|---:|---:|---:|
| square128/original | 128 / 2 | 92.384 | 1487.7 | 82.11% | 72.21% | 87.93% |
| rect144/original | 144 / 2 | 96.160 | 1607.9 | 90.92% | 80.37% | 88.40% |
| aligned148/original | 148 / 2 | 95.616 | 1662.0 | 93.80% | 82.90% | 88.38% |
| aligned148/wide | 148 / 2 | 96.512 | 1646.6 | 94.24% | 82.06% | 87.08% |
| tail152/original | 152 / 3 | 131.072 | 1245.2 | 66.80% | 59.16% | 88.56% |
| aligned592/original | 592 / 8 | 740.448 | 1716.9 | 96.12% | 91.90% | 95.61% |

证据最强的是 `aligned148 → tail152`：M、K 不变，N 从 9472 增至 9728，仅增加 2.70% 算术量；耗时却增加 37.08%。SM active 从 93.80% 降到 66.80%，Tensor/elapsed 从 82.90% 降到 59.16%，但 Tensor/active 仍为 88.38% 和 88.56%。这支持“更多 SM 提前完成，剩余少数任务拖长 kernel”的解释，而不是所有活跃 SM 上的 Tensor Core 都突然变慢。

该 kernel 固定启动 148 个 CTA，组成 74 个双 CTA cluster。每个 cluster 通过 persistent 循环处理多个 `512×256` 逻辑输出任务。148 个逻辑任务恰好两轮；152 个任务需要三轮，第三轮只有 4 个 cluster 有工作。等时任务模型的平均任务槽利用率分别为 100% 与 152/(74×3)=68.47%。这是调度模型，不是实测利用率；实际任务会交错执行，也有启动、写回和动态时钟影响。

`rect144 → aligned148` 的 SM active 从 90.92% 升至 93.80%，Tensor/elapsed 从 80.37% 升至 82.90%，Tensor/active 几乎不变（88.40% → 88.38%），同样支持填满尾轮。`square128 → aligned148` 的 Tensor/active 也接近（87.93% → 88.38%），但矩阵长宽比与缓存复用同时变化，因此不能把全部吞吐差都归到任务数。

`aligned592` 把 M 增至 8192、K 增至 8192，每个 cluster 处理 8 个任务。SM active=96.12%，Tensor/elapsed=91.90%，Tensor/active=95.61%。更长的计算段和更多 persistent 迭代能够摊薄固定开销，但此对照同时改变 M、K、任务数，不能单独证明是哪一个参数贡献最大。这里按 2MNK 得到 1716.9 TFLOPS，仅是标称峰值的 76.3%；与 91.90% Tensor 活跃率不是同一口径。

## 内存与输入流水线

| case / variant | DRAM TB/s | DRAM 峰值利用率 | L2 请求 TB/s | L2 峰值利用率 | L2 sector 命中率 |
|---|---:|---:|---:|---:|---:|
| square128/original | 0.977 | 12.74% | 8.27 | 31.49% | 73.30% |
| rect144/original | 1.241 | 16.19% | 9.61 | 37.01% | 66.94% |
| aligned148/original | 1.290 | 16.82% | 9.98 | 38.66% | 64.59% |
| aligned148/wide | 1.297 | 16.91% | 9.48 | 36.30% | 76.20% |
| tail152/original | 1.003 | 13.07% | 7.03 | 26.30% | 76.20% |
| aligned592/original | 1.378 | 17.96% | 10.42 | 42.79% | 77.56% |

六组 DRAM 整体利用率仅 12.74–17.96%，L2 整体利用率 26.30–42.79%。这些计数器**不支持“全局 HBM 带宽已经跑满”**的判断。它们也不能排除 TMA 请求延迟、局部端口、共享内存访问、barrier 或输入缓冲不够深造成的短暂空档。L2 请求吞吐包括重复请求，不等于矩阵唯一数据量；DRAM 写入量也不必等于 D 的大小，因为写回缓存流量与逻辑输出不同。

在同一 `aligned148` 形状，`wide` 把 epilogue 从四次 64 列改成两次 128 列，同时输入流水线由四级降为三级。其 profile 耗时增加 0.94%，Tensor/active 由 88.38% 降到 87.08%，Tensor/elapsed 由 82.90% 降到 82.06%；SM active 并没有降低（93.80% → 94.24%）。因此这里没有形状尾轮恶化，却有活跃 SM 内的 Tensor 活动小幅下降，符合“减少输入缓冲或改变 epilogue 未带来净收益”的观察。

但 **NCU 本轮没有采集 full_late/full_early，也没有 warp stall、TMA/barrier stall 或按源码归因的采样**。不能据此断言三级/二级流水线的数据等待就是唯一原因，不能给各类 stall 分配百分比。wide 同时增加寄存器占用且带有显式 TMEM fences；要隔离流水线深度，需要同形状、同 epilogue、同 fences 的 stage-only 消融。正式 benchmark 的 full2stage 退化可以作为实验结果报告，不能伪装成 NCU 已直接证明其 stall 来源。

## Launch 与 occupancy

所有 profile 都是 148 CTA × 384 threads，cluster size=2，`launch__cluster_max_active=74`。原始及 full 变体的动态共享内存 225 KiB，加上驱动保留 1 KiB 后是 226 KiB/CTA；wide 为 209+1 KiB。采集的 original 与 wide 均受寄存器和共享内存限制，最多驻留 1 CTA/SM。

| case / variant | registers/thread 实用 / 分配 | 动态共享内存 KiB/CTA | 实测活跃 warp occupancy |
|---|---:|---:|---:|
| square128/original | 94 / 96 | 225 | 18.69% |
| rect144/original | 94 / 96 | 225 | 18.72% |
| aligned148/original | 94 / 96 | 225 | 18.69% |
| aligned148/wide | 161 / 168 | 209 | 18.67% |
| tail152/original | 94 / 96 | 225 | 18.71% |
| aligned592/original | 94 / 96 | 225 | 18.74% |

384 threads=12 warps；相对每 SM 的 64 warp 上限，理论 occupancy 为 18.75%。实测 18.67–18.74%，非常接近理论值。它说明该资源配置下驻留的 warp 数量，不代表只有 18.7% 的 Tensor 算力被使用；warp specialization 和异步 MMA 可以在这种 occupancy 下维持很高的 Tensor 活跃率。wide 的 161 个实用寄存器按分配粒度升至 168，比 original 的 94/96 明显增加，但本轮没有因此从 1 CTA/SM 再降到更少。45 份编译记录均为零 spill。

`launch__waves_per_multiprocessor` 六组都为 1，描述的是**启动网格的物理 CTA waves**。它看不到 persistent kernel 内部 128/144/148/152/592 个逻辑任务需要 2/2/2/3/8 轮。因此 launch waves=1 与逻辑尾轮效应完全可以同时存在。[NVIDIA launch 与 occupancy 指标](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-reference)

## 可支撑的结论

1. 148 对齐负载确实改善整体 SM/Tensor 活跃时间；152 任务制造第三轮只剩 4 个 cluster 的尾部，对吞吐的损害有直接硬件计数器支持。
2. 更长、更大的 592 对齐负载让 Tensor/active 升至 95.61%，说明固定开销及流水线启动/收尾可以被摊薄。
3. wide 没有提高 Tensor 活跃率；现有计数器不足以进一步拆分输入等待、epilogue、寄存器压力或同步成本。未采集的 full2stage 不做 NCU stall 归因。
4. 原始输出路径已经使用 `cp.async.bulk.wait_group.read 0`：等待源共享内存读取完成后复用。不能把它描述成每个 chunk 都等待显存写入彻底完成。[PTX read-wait 语义](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-wait-group)
