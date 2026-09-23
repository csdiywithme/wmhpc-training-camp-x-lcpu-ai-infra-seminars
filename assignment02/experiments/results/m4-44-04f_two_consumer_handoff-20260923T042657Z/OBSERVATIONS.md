# 两段跨 CTA ready 通知：K 复用时错误

从已通过的双 consumer `04d` 派生：warp 3 在各 CTA 发 TMA，等本地 full 后向 CTA 0 的 ready barrier arrive；两个 MMA issue warp 等该 ready barrier。候选在 `512×64×64` 与 `512×64×128` 通过；`512×192×256` 为 `FAIL(bad=40578)`，因此没有测 4096³，也不计性能收益。后续诊断快照 [`04f_two_consumer_handoff_diag`](../m4-44-04f_two_consumer_handoff_diag-20260923T042823Z/run.json) 的首批错误从 row 64 开始，多项 got=0；这尚不足以断定具体的 barrier/proxy 原因。原始源码、构建、输出在 `inputs/` 和 `run.json`。各程序 5 秒超时，无自动重试。
