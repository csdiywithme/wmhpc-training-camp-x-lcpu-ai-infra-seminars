# wide 2048×9472×8192：现有 NCU 报告的逐指令定位

本次只在本地读取已保存的 `.ncu-rep`，没有运行 GPU、重新编译或修改 kernel。
原报告 SHA256：`a38f656aeb22b6d50b310a8de534a32854ff2bb9e52aa2ec456dc5c0212b84be`。
使用本机 Nsight Compute 的 `ncu_report` Python 接口取得 PC correlation IDs、SASS 和实例计数。

**结论：主要采样等待发生在写回 warp 等待 MMA 完成，以及不同职责的线程在退出前会合。普通 shared 指令没有发现地址模式造成的额外 wavefront。另找到 K 循环重复加载固定 TMEM 基址的具体优化候选，但尚无提速证据。**

## 1. Long Scoreboard → 写回等待 Tensor Core 结果

| 位置 | SASS | Long Scoreboard 样本 |
|---|---|---:|
| `0x2b28eb5b2e30` / `+0x37b0` | `@!P1 NANOSLEEP.SYNCS 0x989680` | 6,478 / 7,168 = 90.37% |
| `0x2b28eb5b1850` / `+0x21d0` | `@!P1 BRA …2e20` | 433 |
| `0x2b28eb5b2e50` / `+0x37d0` | `@!P1 BRA …2e20` | 29 |
| 同一 wait 入口和循环合计 | `+0x21c0/+0x21d0/+0x37a0…+0x37e0` | **6,940 / 7,168 = 96.82%** |

这些百分比的分母是该类别的采样数，不是 kernel 墙钟时间。

手工语义对应的证据链（没有编译器行号映射）：

1. wait 检查 `SYNCS.PHASECHK … [R4+0x38]`；R4 包含 `wg_id*8`，即访问 pool 的 `uint64[7+wg_id]`。
2. CUDA 460–462 行初始化的正是这两个 `mma2ld` barrier，字节地址为 `0x38/0x40`。
3. CUDA 565 行 / DSL 381 行在当前输出 tile 的 128 个 K 步提交完后，以 `tcgen05.commit` 通知 `mma2ld`。
4. CUDA **590 行** / DSL **392 行**的 `mma2ld.wait(wg_id, wb_ps.phase)` 等待该结果；SASS wait 返回后开始读 TMEM。

因此可以定位到“写回线程等当前 consumer 的 MMA 完成”。不能把这个 Long Scoreboard 热点直接解释为普通 global load 的 HBM 延迟。
每 CTA 的 12 个 warp 中有 8 个属于写回 warpgroup，它们在 Tensor Core 计算期间等待是预期分工；大量等待样本不意味着存在同等比例可删除的时间。
同时，MMA 的完成也可能包含其上游供数等待，因此定位到等待点并不等于分解了 MMA 内部所有耗时。

`0x989680` 对应 helper 中的 `suspendTimeHint=10000000`，不是每次固定睡眠 10 ms。
phase 完成后可以恢复；参见 [PTX mbarrier.try_wait](https://docs.nvidia.com/cuda/archive/13.1.1/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-mbarrier-try-wait)。

## 2. Barrier / Sleeping → kernel 退出前会合

| 类别 | PC / 偏移 | SASS | 同类样本占比 | 对应位置 |
|---|---|---|---:|---|
| Barrier | `0x2b28eb5b2770` / `+0x30f0` | `UCGABAR_WAIT` | 1,701 / 1,728 = 98.44% | cluster cleanup 同步 |
| Sleeping | `0x2b28eb5b26e0` / `+0x3060` | `WARPSYNC.ALL` | 870 / 933 = 93.25% | 同一个 cluster 同步 helper 的 warp 会合 |

对应 CUDA **678 行** / DSL **426 行**的 `cluster_sync()`，位于各职责分支之后、TMEM dealloc 之前。
提前结束工作的 warp/lane 可在此等待仍然计算或写回的线程；这不是“最后一个计算完成后又串行多等了这么长时间”的证明。
不能凭该采样比例删除同步或断言收益。

## 3. 修正首页 shared-load 2.8-way 警告的解释

| 本报告逐 PC 数据 | 数值 |
|---|---:|
| 有 shared wavefront 数据的指令 PC | 55 |
| `memory_l1_wavefronts_shared` 总和 | 349,354 |
| `memory_l1_wavefronts_shared_ideal` 总和 | 349,354 |
| `derived__memory_l1_wavefronts_shared_excessive` 总和 | **0** |
| 所有普通 LDS 的 wavefront 总和 | **44,326**，等于 Details 的 shared-load 请求数 |

55 个 PC 逐一满足 actual=ideal。写回 `STS.128` 每次正常需要 4 个 wavefront，actual 与 ideal 均为 4；不能把这个宽指令的 4-way 数字直接当成额外 bank conflict。

NVIDIA 在官方论坛解释：Details 的硬件 bank-conflict 计数还包含向高优先级客户端失去仲裁的情况，例如 TMA 填充和 Tensor Core 的 shared 读取；Source 的 excessive 则按指令访问地址、宽度和有效线程分析地址冲突。
参见 [Details 与 Source 指标差异](https://forums.developer.nvidia.com/t/nsight-compute-h100-questions-on-l1-bank-conflict-statistic-discrepancies-between-details-and-source-pages/351780) 和 [Nsight Compute 团队的说明](https://forums.developer.nvidia.com/t/what-does-other-mean-in-shared-memory-tabel/296918)。

**现有证据不支持“修改 epilogue 的 shared 布局即可消除 2.8 路冲突并提速 28.91%”。**
更符合证据的解释是硬件资源仲裁/统计范围差异，但当前报告不能进一步分离具体哪个异步客户端贡献了多少。
结论仅覆盖被 Source 归因的指令，不扩展为“所有异步访存都没有共享内存争用”。

源码也与此吻合：A/B 由 TMA 搬运并由 MMA 直接消费；epilogue 为 TMEM→register→swizzled STS.128→TMA store。
CUDA 595/631 行读取 TMEM，616/650 行写 shared，624/656 行由 TMA 写回；不存在普通 LDS 成片读取 epilogue 数据的路径。

## 4. 具体候选：把固定 TMEM 基址移出 K 循环

| PC / 偏移 | 指令 | 动态执行数 | 地址冲突额外 wavefront |
|---|---|---:|---:|
| `0x2b28eb5b0da0` / `+0x1720` | `LDS R5, [R146]` | 18,944 | 0 |
| `0x2b28eb5b1170` / `+0x1af0` | `LDS R21, [R146]` | 18,944 | 0 |

两条合计 **37,888 次，占普通 shared-load 请求的 85.48%**。
数据流为 `pool[0] → 加 consumer*256 → R2UR → UTCHMMA 的 tmem 参数`。
SASS 对两个 K 步展开，次数恰为 `74 clusters × 2 consumers × 2 tiles × 128 K steps = 37,888`。

生成 CUDA **554–557 行**反复出现 `((uint*)pool_buf_ptr)[0]`；对应 DSL **307–309 行**的 `allocated_addr=tmem_addr[0]` 与 **373–377 行**的 MMA 调用。
该基址在分配完成后至最终释放前保持不变。可检查显式加载成寄存器值、提升到 K 循环外是否能消除这些 LDS。

这是减少重复控制信息读取的候选，不是已确认的主要耗时：两处 PC 仅分别有 2 和 5 个总采样；采样落在消费者处还需追踪依赖，不能仅凭请求数估算时延收益。
本次未实现或运行该候选，也没有更改现有性能结论。

## 5. 流水线边界与既有实验

DSL 367 行的下一 tile MMA 等待 `ld2mma`。当前 wide 在两个 128 列 chunk 均完成 TMEM 读取、shared store、TMA 源读取等待后，才在 421–422 行通知 TMEM 可复用。
这是一条真实的串行依赖，可用于分析计算与写回重叠；报告本身没有给出缩短该边界的可获得收益。
`wait_group.read` 保护的是 TMA 对源 shared 的读取完成，并非全部全局写入完成。

仓库已有 `full_late/full_early` 提前释放对照以及其他 release 候选；不能把“提前释放”当成从未做过的新实验重复运行。
本次没有重新测量任何版本。

## GUI 与证据文件

在 Source → SASS 查看上述完整 PC，或相对本 kernel 第一条 SASS 的偏移；第一条 SASS 为 `0x2b28eb5af680`。
按 `Warp Stall Sampling / Long Scoreboard` 排序，主要定位 `+0x37b0`；共享访存查看 `L1 Wavefronts Shared`、`Ideal`、`Excessive`。

- [带计数注释的 SASS](annotated.sass.txt)
- [逐 PC 数据 CSV](pc_metrics.csv)
- [逐 PC 原始导出 JSON](pc_metrics.json)
- [热点 CSV](hotspots.csv)
- [汇总与校验结果](localization_summary.json)
- [冻结的 DSL 源码](../../runs/20260920T105923Z_78d47693/input/kernels_tuned.py)
- [冻结的生成 CUDA](../../runs/20260920T105923Z_78d47693/input/aligned148_k8192__wide.cu)

导出包括 920 条可用 SASS 和 21 个未附 SASS 的采样 PC；未附 SASS 的位置保留但不强行归类。
全部采样及可加和的已检查 Source 指标，其逐 PC 和均匹配报告总值。
没有 `-lineinfo`，源码位置通过 barrier 偏移、分支目标、寄存器数据流和控制结构手工对应；不能声称是 NCU 自动行号关联。
