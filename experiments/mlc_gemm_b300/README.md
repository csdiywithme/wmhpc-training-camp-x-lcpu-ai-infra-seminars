# MLC GEMM 九步：Modal B300 实测

已完成 2026-09-18 的单卡 B300 实测：[结果与分析](runs/20260918T085349Z/RESULTS.md)、[原始 JSON](runs/20260918T085349Z/artifacts/results.json)、[性能图](runs/20260918T085349Z/performance.png)。九个版本全部通过三种子校验；v9 为 0.09376 ms、1465.83 TFLOPS，达到本次 cuBLAS 吞吐的 95.79%。

本目录是独立的公开教程复现实验，来源为 MLC 的三个相邻章节：

1. [构建 Tiled GEMM](https://mlc.ai/modern-gpu-programming-for-mlsys/zh/chapter_gemm_basics/index.html)：v1–v3。
2. [使用 TMA 为 GEMM 建立 Pipeline](https://mlc.ai/modern-gpu-programming-for-mlsys/zh/chapter_gemm_async/index.html)：v4–v6。
3. [使用 Warp Specialization 和 Cluster 扩展 GEMM](https://mlc.ai/modern-gpu-programming-for-mlsys/zh/chapter_gemm_advanced/index.html)：v7–v9。

上游 revision 为 `ebccca2e5675966f68fb3d4880d4448194bd638d`；本地 `upstream/` 保存阅读依据和 SHA256。实验不修改课程作业的 TODO。

## 版本对应与适配

| 版本 | 本次实现 | 与原文的关系 |
|---|---|---|
| v1 | 同步加载、独立 MMA，单 CTA 串行覆盖完整矩阵 | 原文只计算 128×128×64。本次增加串行输出 tile 循环、K partial 的 FP32 寄存器求和。 |
| v2 | K 循环在 TMEM 累加，单 CTA 串行覆盖完整矩阵 | 原文只计算一个输出 tile。本次增加串行 M/N tile 循环。 |
| v3 | 多 CTA 空间分块 | 保留原文 kernel。 |
| v4 | TMA 异步加载与写回 | 保留原文 kernel。 |
| v5 | 深度为 2 的软件流水线 | 保留原文 kernel。 |
| v6 | Persistent kernel 与 tile scheduler | 保留原文 kernel，SM 数由显式参数提供。 |
| v7 | Warp specialization | 同上。 |
| v8 | 两 CTA cluster | 同上。 |
| v9 | 多 consumer，共享 B tile | 同上。 |

v1/v2 是明确标注的完整矩阵适配版，因此本实验不声称复现原文未公开的完整矩阵 baseline 实现。v1 增加的寄存器归约与重复 TMEM 读取也会影响性能；v1→v2 的提升不能只归因于一条优化。原文 B200 数据和本实验 B300 数据来自不同设备与测试协议，不能当作严格的 GPU 跨代比较。

## 测量口径

- `D = A @ B.T`；A/B 均以 4096×4096 连续行主序储存。
- FP16 输入/输出，FP32 累加；所有版本计算完整输出。
- 每个版本在独立 CPU 进程完成 TIRx 编译和 NVCC cubin 预检查；B300 target 为 `sm_103a`。CPU 导出的共享库保存 CUDA 源码，GPU 加载模块时再次编译，这部分也不计入延迟。
- 同一卡上检查三个随机 seed 的完整输出。参考为禁用 TF32 的 FP32 GEMM 再转换 FP16；统一 `rtol=0.02, atol=0.01`。
- 正确性检查前将输出填 NaN，以发现漏写位置。
- 比较对象为 `torch.mm(..., out=...)` 调用 cuBLAS，关闭 FP16 reduced-precision reduction；算法与 workspace 使用 PyTorch 默认值，无手工调优。
- CUDA events 与所有 kernel 使用同一 CUDA stream；每次测量前写入 256 MiB 缓冲区，清理操作位于计时区间之外。
- 五轮随机化版本顺序；每轮按校准延迟选择 5–200 次调用，按 200 ms 预算计算后取上下限；快速版本受 200 次上限限制，可短于 200 ms。保存全部原始 event 时间。最终值为五个轮均值的中位数。
- 编译、分配、输入生成、验证和缓存清理均不计入延迟。CUDA event 测的是 GPU stream 区间，可能包含 host 提交间隙。
- 保持默认动态频率与功率配置，保存温度、频率、功耗遥测；没有固定频率，因此小幅差距需结合轮间波动理解。

## 运行

在已配置 Modal 的 Python 环境中，从仓库根目录运行：

```bash
MODAL_PROFILE=simidawhu MODAL_DISABLE_API_PROXY=1 \
PYTHONPATH=experiments/mlc_gemm_b300 \
python -m modal run experiments/mlc_gemm_b300/modal_runner.py
```

`--build-only` 只做 CPU 编译；`--mode verify` 只做 GPU 正确性检查；`--reuse-build <run目录>` 复用形状、架构和相关源码 SHA256 一致的构建。

每次实验保存 `runs/<UTC时间>/`，包含请求参数、源码快照、SHA256、构建产物、原始日志与 JSON 结果。Modal 只上传 `FILES` 明确列出的本实验六个脚本，GPU 使用单张 B300，无自动重试，函数超时上限 1500 秒。
