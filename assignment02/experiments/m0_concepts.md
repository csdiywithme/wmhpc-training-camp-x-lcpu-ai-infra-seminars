# 0.3 概念判断讨论记录

2026-09-15。以下区分学员回答与助教 review，未包含 GPU 实测。

学员判断：(a) 对；(b) 对，并询问是否实际上不会有 lane divergence；(c) 错，准备数据的压力更大；(d) 错，可以通过 pipeline 或 reuse 提高利用率。

Review：四个判断正确。(a) 按题面指定的 A/B 读、D 写模型成立。(b) lane-dependent 分支可能让只有部分 lane 执行 MMA，或不同 lane 执行不同次数；`.sync` 不会使这种使用自动合法。此前发生过分支、在 MMA 前已重新汇合，与在发散控制流中执行 MMA 不同。(c) 学员指出供数压力，此外需考虑 fragment 存储与寄存器等资源占用。(d) reuse 可减少每单位计算所需的显存流量，提高 kernel 的显存计算强度；pipeline 可重叠搬运与计算、隐藏等待，但单独不改变 FLOP/byte，不能消除真实带宽上限。

参考：[PTX ISA，mma](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma)。
