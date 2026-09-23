# 2-CTA S=4 深流水尝试：超时

从通过对拍的 `04c_cta_pair.cu` 单独保存 S=4 源码；它让每 CTA 的 A/B stage 从 `3×20 KiB` 增至 `4×20 KiB`，希望用 2-CTA 节省的 B shared 换取更深预取。`sm_100f` 编译成功，但最小 `256×64×64` 在 `timeout -k 2s 5s` 下返回 124，无结果，也没有进入后续 benchmark。原因尚未定位，**不能**记为性能回退或正确版本。源码快照、构建和超时输出在 `inputs/` 与 `run.json`；无自动重试。
