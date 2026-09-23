# 4.4 追加实验：epilogue 与 tile 大小

同次 Modal B300 分配，四版均用 CUDA 13.1、sm_100f、-O2、-lineinfo、STAGES=3。每版四个范围明确的形状严格对拍，再在 4096³ 轮换顺序测三轮。普通程序 5s timeout + 2s grace，NCU 35s + 2s，无自动重试，未锁频。每个程序内包含预热和多次 CUDA event 平均。

本轮 04g 包含 shared scratch 复用前的 `fence.proxy.async.shared::cta`；较早的 04g 初测快照未包含它，初测仅保留作历史，不作为最终同步实现。

| 版本 | 三轮 TFLOPS | 中位数 | 对原 AB | 理论 2250 TFLOPS 达成率 |
|---|---|---:|---:|---:|
| 04ab_warp_persistent | 656.6, 654.4, 654.4 | 654.4 | +0.0% | 29.1% |
| 04g_epilogue | 971.5, 970.1, 968.7 | 970.1 | +48.2% | 43.1% |
| 04h_bn128 | 1138.8, 1139.9, 1140.1 | 1139.9 | +74.2% | 50.7% |
| 04h_bn256 | 957.7, 957.7, 958.3 | 957.7 | +46.3% | 42.6% |

推荐版本：`04h_bn128`（仅以本轮 4096³ 结果选择）。

## 验证范围

- 04ab_warp_persistent: 128×64×64, 128×64×128, 256×192×256, 256×4096×16384，以及三次 4096³，全部 PASS(bad=0)。
- 04g_epilogue: 128×64×64, 128×64×128, 256×192×256, 256×4096×16384，以及三次 4096³，全部 PASS(bad=0)。
- 04h_bn128: 128×128×64, 128×128×128, 256×384×256, 256×4096×16384，以及三次 4096³，全部 PASS(bad=0)。
- 04h_bn256: 128×256×64, 128×256×128, 256×768×256, 256×4096×16384，以及三次 4096³，全部 PASS(bad=0)。

输入为固定 seed 的 BF16 小整数、FP32 累加/输出，对 cuBLAS FP32 输出逐元素严格比较；没有覆盖随机实数、非 tile 整除形状或所有 GPU。BN=128/256 版本分别要求 N 按对应 BN 整除。

## NCU（与普通计时分开）

`--set full --clock-control none --launch-skip 21 --launch-count 1`。原生报告、details/raw/source 导出均在本目录；版本、原始命令及源码/报告 hash 见 run.json。

| 版本 | duration µs | SM throughput % | store requests | store sectors | sectors/request | shared 限制 CTA/SM |
|---|---:|---:|---:|---:|---:|---:|
| 04ab_warp_persistent | 213.15 | 48.79 | 524288 | 16777216 | 32.0 | 3 |
| 04g_epilogue | 142.24 | 71.40 | 524288 | 2097152 | 4.0 | 3 |
| 04h_bn128 | 120.99 | 无效 | 无效 | 无效 | 无效 | 2 |

NCU store sectors 是 L1/TEX 请求的 32B sector 数，不是实际 HBM 写出字节。相同 FP32 输出大小下，sectors/request 降低说明合并写入改善；不能把这些计数直接当成 HBM 流量节省。NCU replay 下主程序打印的 TFLOPS 不参与上面的性能比较。

## 改动与结论

1. `04g_epilogue.cu` 保持 128×64×64、S=3、3 CTA/SM，只改输出路径：`.32x32b.x1` 的 64 次读/等待改成 `.x8` 的 8 次；用已经完成 MMA 读取的 stage storage 作为 FP32 `[128][BN+1]` scratch；线程先按 TMEM 行写入 scratch，再按行主序连续写 global。padding 避免固定列 shared 写入时的 32-way 冲突；CTA barrier 保证重排完成；在下一 tile TMA 复用前执行 proxy fence。没有增加动态 shared 分配，也没有添加 bias/activation，因此这是 epilogue 实现优化，不是算子融合。
2. `04h_bn128.cu` 在上述基础上改 BN=128，将 persistent grid 调到 2 CTA/SM；`04h_bn256.cu` 改 BN=256、1 CTA/SM。BM=128、BK=64、S=3 不变。增大 BN 并不意味着总是更快：BN=128 比 G 快约 17.5%，BN=256 反而比 G 低约 1.3%、比 BN128 低约 16.0%，不选为最佳版。
3. 4096³ 输出 tile 数依次为 2048、1024、512；对应网格 444、296、148 CTA。在均匀 tile 耗时的简化分轮模型下，末轮分配效率约 92.3%、86.5%、86.5%。大 tile 的 tail 并没有更好，性能提升不能归因于填满最后一轮。此模型不是硬件周期级的损失测量。
4. 每 stage 的 FLOP/输入 byte 随 BN 从 42.7→64→85.3 上升，但 shared 容量和驻留数同时变化。BN256 回退与复用、并行度和每 tile 收尾成本之间的权衡一致；没有单独控制所有变量，不能认定回退完全由 occupancy 导致。
5. 本次候选目标为 4096³，不是通用 shape dispatch。`256×4096×16384` 只用于正确性覆盖；不同 BN 会改变并行 tile 数，不能宣称所有形状都加速。

指令布局与等待规则参阅 [PTX tcgen05.ld](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-ld)。详细源码差异另存于 experiments 的三个 `.patch` 文件。用户原 4.3 与旧 AB 均保留。

## BN128 无效计数器的诊断补采

原 full 报告虽然正常退出，但详情里的多数硬件计数为 `NaN`，raw 中部分为 0，不能据此推导利用率。保留异常原件，另在新一次 B300 分配只采六个指标，未重跑性能扫描、未自动重试。

[补采原生报告](../m4-bn128-counters-20260923T045500Z/04h_bn128_counters.ncu-rep)及[原始导出](../m4-bn128-counters-20260923T045500Z/ncu-details.txt)：duration 120.38 µs、SM throughput 57.27%、shared 限制 2 CTA/SM、global store requests 524288、sectors 2097152，即 4 sectors/request。补采同一源码 SHA256 已核对；不同分配，不把该 duration 与上表当成严格同卡配对。

注意 G 的 SM throughput 71.40% 高于更快 H128 的补采 57.27%。SM throughput 是硬件管线利用指标，不能直接当作 `TFLOPS/2250`，也不是越高就保证 GEMM 更快；目标仍以实测耗时为准。
