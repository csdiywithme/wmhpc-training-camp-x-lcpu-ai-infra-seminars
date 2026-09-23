# 4.4 B：persistent CTA（独立于 warp 分工）

[`04b_persistent.cu`](inputs/m4_gemm/04b_persistent.cu)独立从 4.3 派生：最多启动约每 SM 3 个 CTA，每个 CTA 循环处理多个 128×64 输出 tile；TMEM allocation 跨 tile 复用，每个 tile 重新初始化 full/empty barriers，并在复用前等待各 stage 最后一次 MMA 完成。S=3 和 MMA/TMA 主循环仍采用原 4.3 控制流。没有修改用户的 `03_pipeline.cu`。

2026-09-23 Modal B300 SXM6 AC，`sm_100f` 强制编译。四个候选烟测形状和 4096³ 三次均 `PASS(bad=0)`；每个程序 `timeout -k 2s 5s`，无超时或自动重试。完整原始输出、编译命令及源文件哈希见 [`run.json`](run.json)，运行器与上传源码快照均保存在本目录。

| 4096³，TFLOPS | 第 1 对 | 第 2 对 | 第 3 对 | 中位数 |
|---|---:|---:|---:|---:|
| 4.3 S=3 基线 | 473.8 | 474.0 | 474.3 | 474.0 |
| 4.4B persistent | 517.5 | 516.9 | 517.9 | 517.5 |

同卡交替中位数提升 **9.2%**，在本形状上单项有效，保留源码并进一步试与 warp 分工叠加。它的 517.5 TFLOPS 低于 warp 分工单项的 562.0 TFLOPS；后者来自另次分配，选择组合版本仍须同卡直接比较。
