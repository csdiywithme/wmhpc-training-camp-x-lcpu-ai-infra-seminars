# 两侧 TMA 直接报告 CTA 0 barrier：超时

从已通过的双 consumer `04d` 派生，尝试让两 CTA 的 TMA 直接把完成字节计入 CTA 0 的 full barrier，去掉逐 K 的全员 `cluster.sync()`。`sm_100f` 编译成功，但最小 `512×64×64` 在 5 秒超时下返回 124；未进入性能比较。保留原样失败源码和 `run.json`，不能据此判定这种协议本身不可行，只能说明此实现尚未正确完成跨 CTA 交接。
