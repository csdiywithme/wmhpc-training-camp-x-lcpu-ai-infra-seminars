# 04g 初测（历史快照）

同次 B300 分配，四个指定形状与三次 4096³ 严格对拍 PASS。三对 4096³ 中位数：原 AB 655.0 TFLOPS，初版 epilogue 972.9 TFLOPS，+48.5%。

此源码快照尚未包含 shared scratch 复用前的 fence.proxy.async.shared::cta。同步审查后已补齐该 fence，并在 m4-epilogue-tiles-20260923T044929Z 同卡重新验证和测量。**最终选择以复测为准**；保留本次上传快照、编译和运行日志作为过程记录。
