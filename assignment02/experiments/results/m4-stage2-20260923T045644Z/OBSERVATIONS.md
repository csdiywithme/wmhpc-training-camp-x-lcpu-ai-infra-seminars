# 4.4：S=2 与 BK=128 的追加对照

同次 B300，BM=BN=128，epilogue 相同。每版四个形状严格对拍，再轮换顺序测三轮 4096³，全部 PASS(bad=0)。普通程序 5s timeout + 2s grace，NCU 35s + 2s，无自动重试，未锁频。原始命令、编译、设备和源码 hash 见 run.json。

| 版本 | BK | S | stage KiB | 中位 TFLOPS | 对 S3 BK64 | NCU duration µs | shared 限制 CTA/SM | SM throughput % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 04h_bn128 | 64 | 3 | 96 | 1140.7 | 基准 | 120.42 | 2 | 56.45 |
| 04i_bn128_s2 | 64 | 2 | 64 | 1067.2 | −6.4% | 130.05 | 3 | 52.24 |
| 04j_bk128_s2 | 128 | 2 | 128 | 791.3 | −30.6% | 174.46 | 1 | 38.25 |

保留 **04h_bn128.cu（BK64 S3）** 为当前 4096³ 最佳版，I/J 保留为负结果。

**后续补充：I 的负结果仅限上述 4096³。** 在任务可均分的 3072×4736×4096 上，S2=1253.0、S3=1213.2 TFLOPS，S2 反而快 3.3%，六次对拍 PASS。见[尾轮对照](../m4-stage-tail-20260923T045944Z/OBSERVATIONS.md)。因此保留 I 作为形状相关候选，不将其概括为无效优化。

## 实现与解释

- BK64 S2 的 padded epilogue scratch 为 66048 B，比 stage storage 多 512 B；实际动态 shared 按 max(stage,scratch)+1024 B 对齐余量申请，不能机械地只分配 64 KiB。
- BK128 S2 用 3D TMA：global dimensions 为 `{64, M或N, K/64}`，byte strides 为 `{K*2,128}`，box 为 `{64,BM或BN,2}`。shared 是两块 `[BM或BN][64]`，保留 128B swizzle；descriptor 的后半 K 跳过一个 slab。原始 global A/B 数据布局未改，没有 host 转置或预打包。
- BK128 每 stage 发 8 条 K16 MMA，再完成一次 full/empty 协议；K=4096 从 64 个 stage 降到 32 个。每 tile 的 TMA、barrier/commit 次数减少，但 shared 限制仅驻留 1 CTA，仍然回退。不能仅凭同步次数减少就预测加速。
- S2 BK64 将 shared 限制从 2 CTA 提高到 3 CTA，却慢 6.4%。增加驻留数并非充分条件；结果与减少 lookahead 的代价一致，但本轮未采 full/empty 指令级等待，不能量化各因素的贡献。
- 未试 S4；相同 tile 下它占 128 KiB stage storage，预计仅 1 CTA/SM。缺乏供数不足的证据时，不优先继续加深。若进一步定位，应分开看 producer 等 empty 与 consumer 等 full，不能把聚合 barrier stall 都解释为 TMA 慢。
- 简化分轮模型下，S3 BK64 的 1024 tiles / 296 CTA 对应 86.5% 分配效率；S2 BK64 的 1024 / 444 为 76.9%，BK128 S2 的 1024 / 148 为 98.8%。因此这次改变驻留网格也改变了末轮分配，不能将 S2 的回退全部归因于预取深度。BK128 末轮更满仍明显更慢，也说明 tail 不是唯一因素。

## 验证范围与编译

- H：128×128×64、128×128×128、256×384×320、256×4096×16384，以及三次 4096³。
- I：与 H 相同。
- J：128×128×128、128×128×256、256×384×640、256×4096×16384，以及三次 4096³。

输入为固定 seed 的 BF16 小整数、FP32 累加/输出，与 cuBLAS FP32 输出严格逐元素比较。未覆盖随机实数、任意尾块或所有 GPU。

I/J 编译必须显式使用 `make -B ARCH=100f STAGES=2 bin/m4_gemm/04i_bn128_s2` 或对应 J 目标；Makefile 默认 STAGES=3。H 使用 STAGES=3。

## NCU 报告

仅采六个关键指标；三份计数有效，原生报告及 details/raw 导出均已下载，源码和报告 SHA256 已核对。SM throughput 不是 TFLOPS/理论峰值；普通计时和 profiler duration 分开记录。

- [S3 BK64](04h_bn128_4096.ncu-rep) · [详情](04h_bn128-ncu-details.txt)
- [S2 BK64](04i_bn128_s2_4096.ncu-rep) · [详情](04i_bn128_s2-ncu-details.txt)
- [S2 BK128](04j_bk128_s2_4096.ncu-rep) · [详情](04j_bk128_s2-ncu-details.txt)
