# TMEM 基址提升实验结果（2026-09-20）

**循环内的 TMEM 基址读取确实被消除，但这次受控比较没有稳定提速证据。保留原 wide 为默认版本。**

固定 M×N×K=2048×9472×8192，只测试前一轮 NCU 定位得到的新候选。
原 wide 是本次同卡配对控制；旧形状扫描、旧流水线版本和 cuBLAS 调优未重跑。

## 正式性能结果

| 方法 | 块均值的中位时延 | 对应吞吐 |
| --- | ---: | ---: |
| 原 wide | 215.371 µs | 1475.719 TFLOPS |
| TMEM 基址提升 | 214.510 µs | 1481.644 TFLOPS |

12 个配对块的 baseline/hoisted 比值几何平均为 1.003746，即候选收益 **+0.3746%**。
按块 bootstrap 的 95% 描述性区间为 **−3.1389%～+3.9052%**，候选仅 6/12 块更快。
该区间只描述本次 GPU 会话，不代表跨机器、跨会话的总体置信结论。

顺序偏差很强：**12/12 块都是先测的版本获胜**。先测原版的 ABBA 块，候选收益
为 −5.4013%；先测候选的 BAAB 块，候选收益为 +6.5031%。每块第一槽分成
三个连续的 100 次调用段，跨块平均时延依次是 182.803、196.778、214.496 µs。
因此不能把约 0.37% 的差值当作可靠优化，现有遥测也不足以把漂移归因于降频或温度。

本次原版自身约 1476 TFLOPS；历史 1658 TFLOPS 不是同会话、同计时协议的控制组，
不能用两者差距声称候选回退。本次没有重跑 cuBLAS，不能更新与库的相对性能结论。

## 测量和正确性

- B300 SXM6 AC，148 SM，CC 10.3；同卡比较 GPU UUID：
  `84685ad4-9592-56d9-1151-3458135ec60d`。
- 6 个 ABBA 和 6 个 BAAB 块，固定随机顺序；每块 4 槽，每槽 50 次预热和
  300 次正式单调用 CUDA event 样本。两版各 7,200 次，共 **14,400 次**。
- 相同 A/B/D 地址、default stream、无显式清缓存、动态时钟；计时不含校验、
  编译和预热。单调用 event 区间可能包含主机提交间隙。
- FP16 输入/输出、FP32 累加。3 seeds 的全输出参考检查、计时后的两次复验全部通过
  （atol=0.01，rtol=0.02）；两版在 3 seeds 的 **58,195,968 个输出元素逐位一致**。

原始数据：[benchmark/results.json](runs/hoist_abba_v2/benchmark/results.json)。
独立重算：[independent_benchmark_audit.md](runs/hoist_abba_v2/independent_benchmark_audit.md)。

## 修改实际产生了什么

在 MMA elected lane 中、persistent tile 循环之前，读取一次 `tmem_addr[0]` 到
显式标量，用它构造 MMA 的 TMEM view。TMA producer、写回、释放及同步逻辑保持原样。
有效输出 tile512×256、K tile64、3-stage、写回块128列、148 CTA / 74 cluster 不变；
148 个逻辑输出任务仍由每个 cluster 处理两轮。

| 检查项 | 原 wide | 候选 |
| --- | ---: | ---: |
| K 循环 SASS 中静态 LDS | 2 | 0 |
| 每线程寄存器 | 161 | 161 |
| 每 CTA 动态共享内存 | 214,016 B | 214,016 B |
| spill / stack | 0 | 0 |
| 编译器 K 循环展开步数 | 2 | 4 |
| 每输出任务、每 consumer 的动态 MMA | 512 | 512 |

**编译器同时改变了循环展开，因此这是一个源代码改动的对照，不是只少两条 SASS
而其余机器码完全相同的对照。** 不能将性能差值全部归因于 LDS。
详见 [build_summary.json](runs/hoist_abba_v2/build/build_summary.json)。

## NCU 核验和本地 GUI 文件

首次全 section 采集在 34 passes 后出现 `LaunchFailed` / Xid 43，未完成输出校验。
其报告保留用于排错，**不作为有效性能或正确性证据**。错误根因尚未确定，不能直接
归为 NCU 缺陷，也不能仅凭无 profiler 时通过校验就排除潜在同步问题。

随后仅重试失败的采集，缩减为 8 项指标、3 passes，没有重跑正式测速。
该采集正常退出，完整 19,398,656 个输出元素校验通过；采集时使用的是另一张 B300
（UUID `30057c84-74e2-87f8-b27d-b04bb55ddf37`），候选二进制哈希与正式测速完全相同。
重试启动曾因远端导入路径错误在初始化阶段失败，修正前没有执行目标采集；该失败
调用已停止并保留 [记录](runs/hoist_abba_v2/profile_retry_hardware/initialization_failed_invocation.json)。

| NCU 指标 | 历史有效 wide 报告 | 本次有效候选报告 |
| --- | ---: | ---: |
| 普通 shared-load 指令 `smsp__sass_inst_executed_op_shared_ld.sum` | 44,326 | **6,586** |
| 普通 shared-store 指令 | 76,368 | 76,368 |
| registers/thread | 161 | 161 |
| dynamic shared bytes/CTA | 214,016 | 214,016 |
| grid CTAs × threads/CTA | 148 × 384 | 148 × 384 |

普通 shared-load 指令减少 **85.1419%**：44,326 − 37,888 个原循环加载 + 148 个
提升后的加载 = 6,586，与静态代码预测一致。这不表示全部共享内存流量减少 85%，
该指标也不包含所有 TMA / Tensor Core 对共享内存的访问。

候选诊断值为 173.44 µs、Tensor pipeline elapsed 利用率 88.95%。这些值来自
profiler 回放，并非正式无 profiler 测速；历史基线来自另一会话，**不将 NCU 时延差
用作配对提速证据**。本次仅补采这些指标，没有有效的新 WarpStateStats/SourceCounters
全报告，因此也不能声称等待热点占比已经改善。

可在本地 Nsight Compute GUI 打开：
[candidate_hardware.ncu-rep](runs/hoist_abba_v2/profile_retry_hardware/candidate_hardware.ncu-rep)。
这是精简指标报告，GUI 不会包含第一次完整报告里的所有分析 section。

- [NCU 导出详情](runs/hoist_abba_v2/profile_retry_hardware/details.txt)
- [NCU 指标对照和报告哈希](runs/hoist_abba_v2/ncu_comparison.json)
- [成功采集状态与完整命令](runs/hoist_abba_v2/profile_retry_hardware/status.json)
- [采集后正确性检查](runs/hoist_abba_v2/profile_retry_hardware/profile_candidate/profile_target.json)

## 结论的范围

实验确认减少指令这一机制成立，但没有确认它能提升端到端吞吐。就当前证据，循环里
的 TMEM 基址 LDS 不是已证实的主要瓶颈；删除大量轻量指令，不必然缩短 Tensor MMA、
TMA 供数和同步所决定的执行时间。这个解释是与数据相符的推断，不能替代更细的因果定位。

后续若继续，应先解决测试中的顺序漂移，再评价更小的收益；本次不自动扩大实验范围。
原始父级 `status.json` 保留 `failed`，忠实记录全 section NCU 的失败。正式 benchmark
和精简 NCU 分别有独立的 `complete` 状态，不修改原始记录掩盖失败。
