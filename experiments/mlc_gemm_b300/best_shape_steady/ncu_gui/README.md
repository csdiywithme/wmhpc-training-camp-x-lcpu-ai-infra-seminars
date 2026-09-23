# wide 2048×9472×8192 的 Nsight Compute GUI 报告

已下载并验证的报告：
[wide_M2048_N9472_K8192_sections.ncu-rep](/Users/simida/csdiy/wmhpc-training-camp-x-lcpu-ai-infra-seminars/experiments/mlc_gemm_b300/best_shape_steady/ncu_gui/runs/20260920T105923Z_78d47693/wide_M2048_N9472_K8192_sections.ncu-rep)。

报告 517,620 字节，1 个 kernel，13 个标准分析 section 加命令行 Tensor Core
计数器。原始 CSV 含元数据共 1603 列，无 NaN/Inf；19,398,656 个输出元素全部
校验通过。采集端 NCU 为 2025.4.1，本地 GUI 为 2026.3.1。

SHA256：`a38f656aeb22b6d50b310a8de534a32854ff2bb9e52aa2ec456dc5c0212b84be`。
同目录 `local_audit.json` 记录独立本地验收，`details.txt`/`metrics.csv` 可离线查看。

仅补采用户选中的内核/形状，不重新运行性能 sweep。此前唯一 wide 报告的
K=4096；当前 K=8192 的报告是新采集。

复用 `best_shape_steady/runs/20260920T051356Z/build/aligned148_k8192__wide.so`，
SHA256 为 `292e320fc1373aa69b4df5e9ebaef849300c740774f8dfc11099ca464439b683`。
不改变 tile、流水线、编译选项；TVM 加载库时仍按原流程编译嵌入的 CUDA 模块。

- 形状 M=2048、N=9472、K=8192，运算 D=A@B.T。
- FP16 输入/输出、FP32 累加；完整 FP32 参考校验。
- 有效输出 tile 为 512×256，148 个任务，由 74 个双 CTA cluster 各处理两块。
- 同一缓冲区预热 50 次，无显式清缓存；NVTX 范围内仅一次目标调用。
- 最终采集显式选择 13 个分析 sections，包含 Compute、Memory、Scheduler、
  Warp State、Instruction、Launch、Occupancy、Source Counters、SOL 等；
  `--replay-mode kernel --cache-control none --clock-control none`。
- 单次应用调用会被 NCU 重放以收集不同计数器，不能把 NCU duration 直接替换
  正常运行的 191.69 μs，也不能视为复现 1658.06 TFLOPS 的普通测速。
- 不给原版本新增 `-lineinfo`。报告可查看 SASS，CUDA 源码行号关联以实际可用
  信息为准；随附原生成 `.cu`、`.tirx.py` 和生成器源码。

`modal_profile.py` 将状态、报告、校验结果和输入文件持久化到
`mlc-b300-wide-ncu-results` Volume。GPU 自动重试关闭，已有运行禁止再次启动。
本地 `runs/wide_2048x9472x8192_full/invocation.json` 保存远端 run ID。
若客户端返回中断，只下载该 run ID 的文件，不再次执行采集。

用 Nsight Compute 的 **File → Open** 打开 `wide_M2048_N9472_K8192_sections.ncu-rep`。
Details 页查看各分析 section，Source 页查看指令；导出的 `details.txt`、
`metrics.csv`、`session.txt` 可用于不启动 GUI 的离线核验。

参考：[NVIDIA Nsight Compute CLI 文档](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)。

## 采集记录与限制

首次 `runs/20260920T105459Z_709cc877` 使用 `--set full`，7 次 replay 后多数
聚合硬件计数器为 NaN。文件保留用于排错，不作为有效性能证据或推荐 GUI 报告。

最终 `runs/20260920T105923Z_78d47693` 使用显式 sections，37 次 replay 后
核心计数器有效：SM active 95.54%、Tensor/elapsed 87.76%、Tensor/active 91.86%、
L2 hit 75.74%。NCU duration 为 175.328 μs，仅作诊断。6 个 CTC 指标仍提示不可访问，
不影响上述核心指标；不能由两次采集推断出首次异常的唯一原因。

最终 run 的原始 `status.json` 保留 `failed`：原因是远端验收误要求该次没有请求的
`dram__throughput.avg.pct_of_peak_sustained_elapsed`，并非 DRAM 采集失败。
实际 section 使用 `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed`，值为
20.25%，且 DRAM cycles/bytes 均有效。本地独立审计已更正这一误报，无需重跑 GPU。

## 逐指令离线定位

[定位报告](analysis/20260920T105923Z_78d47693/LOCALIZATION.md) 根据原 `.ncu-rep`
的 PC correlation IDs 和 SASS，将 96.82% 的 Long Scoreboard 样本对应到写回等待
MMA 完成的同一 barrier 路径；主要 Barrier/Sleeping 样本位于退出前的 cluster 会合。
55 个 Source shared 指令 PC 的 actual/ideal wavefront 完全一致，excessive 为零，
因此不能把 Details 的 2.8-way 硬件提示直接解释为 epilogue 地址冲突。
另记录了 K 循环重复加载固定 TMEM 基址的候选；没有实现、测试或声称提速。

`export_pc_analysis.py` 仅使用本地 NVIDIA `ncu_report` 接口读取已有文件；
分析目录保存逐 PC JSON/CSV、带计数注释的 SASS 与热点表，不使用 GPU。
