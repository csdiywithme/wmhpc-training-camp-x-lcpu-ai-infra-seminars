# M4 4096³ 梯子：同一次 B300 分配

2026-09-23，NVIDIA B300 SXM6 AC，驱动 580.95.05，CUDA 13.1。一次 Modal `run()` 内顺序运行 Assignment01 naive FP32（原 kernel、默认 BS=16）和 Assignment02 4.1/4.2/4.3（BF16，4.3 为 S=3）。四个程序均从本次上传源码编译，全部 `PASS(bad=0)`，退出码 0；每个程序分别设 `timeout -k 2s 5s`，无超时或自动重试。所有构建命令、程序原始输出、GPU 查询及远端/本地源码 SHA256 核对见 [`run.json`](run.json)，实际输入和运行器保存在 [`inputs/`](inputs/) 与 [`modal_m4_ladder.py`](modal_m4_ladder.py)。

| 实现 | 精度 | 4096³ 普通运行 | TFLOPS | 同程序 cuBLAS TFLOPS |
|---|---|---:|---:|---:|
| Assignment01 naive，BS=16 | FP32 | 21.659 ms | 6.3455 | — |
| 4.1 tiled | BF16/FP32 累加 | 3.45 ms | 39.8 | 1762.2 |
| 4.2 TMA 单缓冲 | BF16/FP32 累加 | 约 0.26 ms | 519.9 | 1764.6 |
| 4.3 pipeline，S=3 | BF16/FP32 累加 | 约 0.29 ms | 469.8 | 1764.6 |

naive 直接调用原 `assignment01/cuda/bonus/matmul.cu` 的 `matmul_naive` kernel，只有测量外壳另写：原程序的 host `main` 固定为 1024³，且会跑耗时的 CPU 全量参考，不适合此表。外壳设 4096³、A/B 均为精确的 `1/128`，核验所有输出恰为 `0.25`，随后对两次 kernel 调用做 CUDA event 平均计时。这个输入和两次计时只支持性能量级比较，不能代表随机输入下的稳健统计。FP32 naive 与 BF16 Tensor Core 后三行不是同精度优化对照；不能把它除以 BF16 cuBLAS 当成同精度达成率。

同一分配下 4.2 相对 4.1 约 13.1 倍；4.3 S=3 比 4.2 低约 9.6%。三行 BF16 与各自程序打印的 cuBLAS 对照约为 2.26%、29.46%、26.62%。程序打印的毫秒数保留位数较少，以打印的 TFLOPS 作比例计算。本次是一次未锁频的普通运行，差异的定性解释还需结合此前 NCU 报告；不是 NCU replay 时间。
