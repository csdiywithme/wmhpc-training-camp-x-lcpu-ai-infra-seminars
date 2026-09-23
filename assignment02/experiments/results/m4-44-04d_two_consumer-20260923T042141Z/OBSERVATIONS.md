# 4.4：两条 group2 MMA consumer 共享 B

在 2-CTA 初版上增加第二个 consumer：CTA 0 的 warp 0、warp 1 各由 lane 0 发起一次不同 A 行的 group2 MMA；每 CTA 同一份 B tile 被复用两次。两个 accumulator 使用不重叠的 TMEM 列区间，stage 的 empty barrier 等两条 MMA 的完成通知。cluster 输出 tile 从 `256×64` 扩为 `512×64`。为控制 shared 用量，本版固定 `STAGES=2`；它仍保留每 K 轮全员 `cluster.sync()`。

B300 SXM6 AC、`sm_100f`，`512×64×64`、`512×64×128`、`512×192×256`、`512×4096×16384` 四个候选用例均 `PASS(bad=0)`。4096³ 同次分配交替三对：

| TFLOPS | 第 1 对 | 第 2 对 | 第 3 对 | 中位数 |
|---|---:|---:|---:|---:|
| 单 consumer 2-CTA | 280.0 | 280.2 | 280.5 | 280.2 |
| 双 consumer 2-CTA | 384.1 | 384.2 | 385.6 | 384.2 |

双 consumer 相对单 consumer **+37.1%**，但仍低于同形状 4.3 单 CTA 基线约 476 TFLOPS，所以保留为实验中间版，不作为最终最佳。对比同时改变了每簇输出 tile、stage 深度与 B 复用，不能把整项增益单独归因于一个因素。构建、输出、源码 SHA256 在 `run.json` 和 `inputs/`；每程序 `timeout -k 2s 5s`，无重试。
