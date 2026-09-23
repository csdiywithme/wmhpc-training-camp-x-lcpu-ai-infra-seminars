# 0.1：B300 最小 MMA 实验记录

状态：B300 的编译、运行和产物观察已取得证据；学员解释待写。程序为课程原版，由助手调用用户已授权的 Modal 工作区执行；本次没有个人 CUDA kernel 实现或性能优化。

## 环境与追溯

- GPU：`NVIDIA B300 SXM6 AC`。
- 驱动：`580.95.05`；nvcc：CUDA `13.1`、`V13.1.80`。
- 镜像：`nvidia/cuda:13.1.0-devel-ubuntu24.04`；g++ `13.3.0`。
- 成功运行时间：2026-09-13 06:56:52–06:56:54 UTC（北京时间 14:56:52–14:56:54）。
- 测前、测后 `nvidia-smi --query-compute-apps=pid,process_name --format=csv` 均返回成功，输出仅表头。
- [Modal 本轮记录](https://modal.com/apps/csdiywithme/main/ap-mV5QzuXpqEMsIZiZTgrOzD)。运行已结束。
- [运行原始数据](run.json)、[编译与 fatbin 列表](build.json)、[本地版本与源码状态](local-metadata.json)、[当次运行器](modal_m0.py)。
- 实际执行的课程输入快照位于 `inputs/`；`build.json` 和 `run.json` 中保存 SHA256。
- 已独立复核：保存产物、输入、PTX、运行器快照的 SHA256 均一致；三个课程输入与记录的 Git 提交逐字节相同；11 条构建命令全部退出 0，运行侧除 120a 返回 1 外其余命令均退出 0，无 timeout/OSError。

## 实际命令与结果

以下编译发生在 CPU 镜像构建阶段，工作目录为 `/opt/m0/cuda`。两个编译结果随即分别改名为 `first_mma_100f`、`first_mma_120a`，避免互相覆盖；完整命令及退出码见 `build.json`。

| 架构 | 编译命令 | 编译退出码 | B300 运行退出码 | 观察 |
|---|---|---|---|---|
| 100f | `make -B ARCH=100f bin/m0_env/01_first_mma` | 0 | 0 | 输出 `PASS` |
| 120a | `make -B ARCH=120a bin/m0_env/01_first_mma` | 0 | 1 | `cudaErrorNoKernelImageForDevice` |

100f 的完整 stdout：

```text
D[0][0]=2 D[0][7]=2 D[15][0]=2 D[15][7]=2
PASS
```

120a 的完整 stderr：

```text
CUDA error cudaErrorNoKernelImageForDevice at m0_env/01_first_mma.cu:76: no kernel image is available for execution on the device
```

报错位置是 [源码快照](inputs/m0_env/01_first_mma.cu) 第 76 行的 `CUDA_CHECK_KERNEL()`，位于第 75 行 `mma_demo<<<1, 32>>>` 之后。原始输出没有矩阵对拍失败记录，也没有 timeout 或操作系统启动错误。

## 编译产物观察

- 两次编译分别使用显式 `-gencode arch=compute_100f,code=sm_100f` 与 `-gencode arch=compute_120a,code=sm_120a`。
- `cuobjdump --list-elf` 对 100f 列出两个名字带 `sm_100.cubin` 的条目，对 120a 列出两个名字带 `sm_120a.cubin` 的条目。这里按工具原样记录名称。
- 两个 binary 的 `cuobjdump --list-ptx` 均退出 0，stdout 为空，stderr 报告 `No PTX file found to extract ...`；完整原文在 `build.json`。
- 另外通过 Makefile 的 `ptx/m0_env/01_first_mma` 目标导出了 [独立 PTX](first_mma_100f.ptx)，其头部为 `.version 9.1`、`.target sm_100f`、`.address_size 64`。
- 该 PTX 第 88 行包含 `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`。这份独立导出文件与 binary 内部代码列表分别存证，后续解释时要区分二者。

## 运行器修正历史与记录限制

首轮 [m0-20260913T065439829799Z](../m0-20260913T065439829799Z/) 在 GPU 函数收集结果末尾，因读取 Modal 包版本元数据抛出 `PackageNotFoundError`，未返回 `run.json`，因此不能用首轮证明 CUDA PASS。首轮错误和当时运行器已保留。

随后修正辅助脚本：包版本元数据缺失时记录 `null`；每条命令执行后立即把原始记录写入 Modal 日志。未修改课程源文件。第二轮使用已构建的镜像和相同输入重新采集，成功返回本目录数据；因此 `build.json` 的构建时间早于本轮 GPU 运行时间。两轮函数的自动重试配置均为 0，GPU 函数上限均为 120 秒。

本地 Modal CLI 为 `1.5.5`。远程 `remote_modal_version` 为 `null`，表示容器没有相应 distribution 元数据，不代表已查明远端版本号。

`run.json` 的 `elapsed_seconds` 统计的是整个子进程的墙钟时间，包含初始化等开销，不能填为 kernel 延迟或据此计算 TFLOPS。0.1 的原始程序只做正确性验证。本次未运行 5090，也未测 Tensor Core 性能峰值。

## 待学员完成的解释

1. 两个 ARCH 都成功编译，但在同一块 B300 上运行结果不同。请结合实际编译参数、错误名称以及 A01 fatbin/JIT 的知识解释。
2. 单独导出的 `.ptx` 文件，和可执行文件中可供运行时使用的代码，是什么关系？用这次的两个产物观察支持你的说法。
3. 找到 PTX 的 MMA 行，对照刚才阅读的 fragment 文档，说明这次运行使用的 shape、A/B 类型与累加类型。

学员解释（2026-09-15）：用户提出“120a 不包含 B300 吧”。

Review：架构不匹配的判断正确。[NVIDIA 官方表](https://developer.nvidia.com/cuda/gpus)列出 B300 的 compute capability 为 10.3，RTX 5090 为 12.0。当前已确认这一点；关于编译成功的含义、fatbin/PTX/JIT 和实际 MMA 类型的说明仍待学员补充。
