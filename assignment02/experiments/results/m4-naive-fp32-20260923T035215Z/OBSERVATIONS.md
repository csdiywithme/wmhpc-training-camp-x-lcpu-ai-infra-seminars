# Assignment01 naive FP32 的独立预跑

2026-09-23 在 Modal B300 SXM6 AC 上编译原 `assignment01/cuda/bonus/matmul.cu` 的 `matmul_naive` kernel（默认 BS=16），独立测量外壳将形状改为 4096³。A/B 均为 `1/128`，全部输出检查为 `0.25`，`PASS(bad=0)`；两次 CUDA event 计时平均 **21.634 ms，6.3529 TFLOPS**。运行命令 `timeout -k 2s 5s ./naive4096` 正常退出，无自动重试。构建、GPU 查询、源码 SHA256 及原始输出见 [`run.json`](run.json)，代码快照在 [`inputs/`](inputs/)。

此为预跑；同一次 B300 分配内与 4.1/4.2/4.3 顺序运行的正式梯子见[后续记录](../m4-ladder-20260923T035429Z/OBSERVATIONS.md)。两次 B300 分配不能视作同一物理卡。本实验只有两次计时、输入固定且未锁频，只用于性能量级。
