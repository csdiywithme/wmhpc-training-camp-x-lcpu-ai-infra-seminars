# B300：任务形状与 GEMM 流水线对比

这是相邻目录九版本复现的后续实验。目标是检验持久化 cluster 调度的尾轮效应，并比较输出写回与下一轮矩阵计算的重叠。

## 实验维度

运算为 `D[M,N] = A[M,K] @ B[N,K].T`，FP16 输入和输出、FP32 累加。逻辑输出块为 `512×256`，148 个常驻 CTA 组成 74 个双 CTA cluster worker。因此 148 个**逻辑任务**对应每个 worker 两个输出块，CUDA grid 并没有变成 296 个 CTA。

`request.json` 包含九种形状：128 个任务的方阵及等 FLOP 长方阵，144/148/152 个任务的相邻形状，296/592 个任务的整轮形状，以及固定 148 个任务时增大 K 的两种形状。

| 版本 | 输入流水线深度 | 每个 consumer 输出 tile 的 TMA store 次数 | TMEM 释放边界 | 共享内存/CTA |
|---|---:|---:|---|---:|
| original | 4 | 4，每次 64 列 | 四次 TMA 读取源共享内存完成后 | 225 KiB |
| fenced | 4 | 4，每次 64 列 | 同 original，补显式跨线程 TMEM fences | 225 KiB |
| wide | 3 | 2，每次 128 列 | 两次 TMA 读取源共享内存完成后 | 209 KiB |
| full_late | 2 | 1，每次 256 列 | TMA 读取源共享内存完成后 | 225 KiB |
| full_early | 2 | 1，每次 256 列 | TMEM 全部读入并存入共享内存后 | 225 KiB |

三个流水线候选均带相同显式 fences。`full_late` 与 `full_early` 仅改变累加器释放边界，是提前释放的直接消融实验。其他跨候选比较同时改变了写回宽度和输入流水线深度。

这里的 `T.ptx.cp_async.bulk.wait_group(0)` 经 TVM 默认 lower 为 `cp.async.bulk.wait_group.read 0`：等待已提交 TMA store 对源共享内存的读取完成，以便安全复用共享内存，并非等待全局内存写入全部完成。`full_early` 在发起 TMA store 前释放 TMEM 累加器，但仍通过该 read-wait 保护 Dsmem 的复用。

## 已完成的实测

[20260918T092500Z 完整报告](runs/20260918T092500Z/REPORT.md) 包含九种形状、六种实现的完整校验与计时。最佳流水线候选为 `wide` 在 `2048×9472×8192` 上的 1656.41 TFLOPS；它相对同形状 `fenced` 的优势仅约 0.52%，低于双方轮间 CV 的波动尺度，本次未证明确切的流水线收益。`fenced` 只是显式 TMEM fence 的同步控制组，不计作流水线优化。

六组 Nsight Compute 硬件采集及 CSV 导出实际全部成功；原始 summary 的 `failed` 是旧解析器不识别 wide CSV 的误报。原始文件保留不变，恢复结果见 [NCU_ANALYSIS.md](runs/20260918T092500Z/NCU_ANALYSIS.md) 和 [ncu_metrics.json](runs/20260918T092500Z/ncu_metrics.json)。

## 运行

在已认证的 Modal 环境，设置 `MODAL_PROFILE=simidawhu`、`MODAL_DISABLE_API_PROXY=1` 和 `PYTHONPATH=experiments/mlc_gemm_b300/tuning`，运行：

```sh
python -m modal run experiments/mlc_gemm_b300/tuning/modal_runner.py
```

CPU 阶段编译所有内核并用 `nvcc --cubin -arch=sm_103a` 检查生成代码。只有全部成功才分配一张 B300。GPU 阶段先验证、后计时，最后单独调用 Nsight Compute。源码、SHA256、编译记录、原始样本和 profiler 文件保存在 `runs/<UTC时间>/`。

每种形状、每个版本均用三个随机种子检查完整输出；计时后再检查反复执行后的输出。性能采用同一 stream 上的 CUDA events，每次计时前在事件外清刷 256 MiB 缓存，五轮随机版本顺序，报告每轮均值的中位数。cuBLAS 使用同形状和同计时方法。

Nsight Compute 使用 NVTX 精确选择一次目标内核，动态查询可用指标，保存 `.ncu-rep`、CSV 和日志。其回放计时与上述 benchmark 分开；若宿主机禁止硬件计数器，记录不可用原因。

```sh
python experiments/mlc_gemm_b300/tuning/report.py RUN_DIRECTORY
```

报告脚本需要 matplotlib；`--no-plot` 可仅生成文本和 CSV。

## 解释边界

相同形状内的延迟比可用于判断流水线改动。不同形状的 TFLOP/s 比是不同工作负载的吞吐比较，不能解释成完成同一任务的加速比。静态 wave 模型假设各 tile 耗时相同，并非实测 Tensor Core 利用率；长宽比、缓存复用、K 长度及 GPU 动态时钟都会影响实际结果。

参考：[MLC 教程](https://mlc.ai/modern-gpu-programming-for-mlsys/zh/chapter_gemm_advanced/index.html)、[PTX tcgen05 内存模型](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-memory-consistency-model)、[PTX read-wait 语义](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-wait-group)、[Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)、[NVIDIA 计数器权限说明](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters)。
