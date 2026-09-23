# B300 新指令探索：首版结果备忘

以下保留2026-09-16首版结果。2026-09-17本阶段已完成收尾；六个C1、六个C2变体及最终独立验证的完整结论见 [STAGE_REPORT.md](STAGE_REPORT.md)。本文件中的“正在”描述实验当时状态，后续结果以阶段报告及JOURNAL为准。

## 第一版结论

两条首版均把 tcgen05 接入完整计算链，并通过对应正确性验收；性能则全面落后于原版。因此，首版不能作为性能优化交付。它们证明了数值语义可以保持，也暴露了数据搬运、布局和中间结果处理成本，需要进一步对照验证。

| 项目 | 正确性实测 | 完整链速度比：原版延迟/新延迟 | 状态 |
|---|---|---|---|
| C1 fused K2 | 旧30组+8接口，38/38输出及末state位等值 | 6形状0.06328–0.08644，几何平均0.07175 | 正确但显著变慢 |
| C2 transposed partial + feature merge | 冻结full：baseline168/168、新candidate168/168；新backend覆盖168、fallback0 | 32组0.04386–0.14397，几何平均0.09399；相对旧merge-only为0.08407 | 正确但显著变慢 |

这里的速度比小于1表示更慢。C1是包含binding/beta准备的完整forward CUDA event计时；C2是热地址CUDA Graph的partial+merge链。两者不能互相比绝对时延，也不是服务端到端吞吐。

## C1 首版证据

- [完整验收](runs/20260916T155241003026Z-c1-verify/artifacts/verify-fused.json)：原官方舍入baseline前置通过，保留独立FP32 naive数学误差。位等值是相对旧实现，不是数学误差为零。
- [完整性能及全部样本](runs/20260916T155457579434Z-c1-bench/artifacts/bench-fused.json)：H12/H96 × 单序列8192、6段varlen、8×1024，共6形状，5轮AB/BA。已计时路径在计时外复验。
- [构建日志](runs/20260916T154757438758Z-c1-smoke/build.json)：K2 174寄存器、0spill；实际目标sm_103a。
- [实际SASS](runs/20260916T154757438758Z-c1-smoke/artifacts/binary-0.sass)及同目录resource JSON。

| H | 输入组织 | 原版µs | 新版µs | 速度比 |
|---:|---|---:|---:|---:|
| 12 | 1×8192 | 835.14 | 13198.09 | 0.06328 |
| 12 | 6段varlen，合计8192 | 356.76 | 5196.25 | 0.06866 |
| 12 | 8×1024 | 154.03 | 1781.99 | 0.08644 |
| 96 | 1×8192 | 1076.99 | 14069.77 | 0.07655 |
| 96 | 6段varlen，合计8192 | 886.53 | 12543.53 | 0.07068 |
| 96 | 8×1024 | 699.34 | 10411.35 | 0.06717 |

H96/8192禁用K1后的诊断调用为13782.85µs，指向K2路径为主。该调用仍含binding准备，不称纯kernel duration。共享中间结果区、bank冲突和布局转换是当前假设；寄存器spill已被编译证据排除。正在实现直接TMEM epilogue与共享布局消融。

## C2 首版证据

- [冻结full验收](runs/20260916T155241079346Z-c2-verify/artifacts/heldout.json)：原protocol digest及family bounds未更改。
- [逐调用覆盖审计](runs/20260916T155241079346Z-c2-verify/artifacts/adapter_calls.jsonl)：168条均执行新partial，无fallback。
- [三条完整链的全部性能样本](runs/20260916T155513744579Z-c2-bench/artifacts/paired.json)：2TP×4batch×2storage×2seed，7轮随机顺序，约10ms目标区间；计时前后使用冻结family bounds复验。
- [构建日志](runs/20260916T154921959378Z-c2-smoke/build.json)：两个storage分支均70寄存器、0spill；BF16 MMA，FP8仅为存储兼容与反量化。
- [实际SASS](runs/20260916T154921959378Z-c2-smoke/artifacts/binary-0.sass)。

代表输入TP1/B1/BF16/seed101：原版5.682µs，旧merge-only5.057µs，新版43.828µs；TP1/B16同族则是15.709、13.589、356.839µs。完整32组全部退化，不仅是少数异常点。正在用NCU观察访存合并、shared bank conflict、同步与TMEM消费，再进行显式供数/布局版本对照。

## 后续完成情况

C1后续五个变体均完成38/38与六形状计时，最终wide-rowmajor为0.120046×，新seed20260917长链4/4逐位一致。C2后续五个变体均完成168/168与32组计时，首版仍为最高样本几何均值0.093990×；最终独立192条也通过。所有变体均未超过旧实现。

C2 [首版NCU](runs/20260916T160153199885Z-c2-profile/artifacts/profile-selected.json)显示shared wavefront实际/理想约4.106倍、额外413696，global load requests70400，零spill。Profiler耗时不代替正式benchmark。机制解释及限制将独立记录。

B200尚未运行，不能声称跨架构复现。当前没有服务端到端、PDL新后端或原生FP8 MMA收益结论。所有启动、编译失败和后续修复详见[JOURNAL](JOURNAL.md)，假设及区分实验见[HYPOTHESES](HYPOTHESES.md)。
