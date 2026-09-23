# 4.4 分步优化记录

**最新选择（2026-09-23）：[`04h_bn128.cu`](../cuda/m4_gemm/04h_bn128.cu)，BM=BN=128、BK=64、S=3。** 在旧 AB 上先优化 epilogue，再增大 BN；中间版本全部保留。S2 和 BK128 S2 的负结果也保留。以下较早实验的数值保持原样，不跨次分配拼接收益。

这里的最新选择针对 4096³。[888 个输出 tile 的均分对照](results/m4-stage-tail-20260923T045944Z/OBSERVATIONS.md)中，I（S2）=1253.0、H（S3）=1213.2 TFLOPS，S2 快 3.3%。保留 I 作为形状相关候选；stage、驻留数和 tail 必须一起分析。

| 追加版本 | 改动 | 同卡 4096³ 中位 TFLOPS | 选择 |
|---|---|---:|---|
| AB | 本轮参照 | 654.4 | 保留历史 |
| [G](../cuda/m4_gemm/04g_epilogue.cu) | x8 TMEM 读回、复用 shared 重排、合并写出 | 970.1 | 有效，+48.2% |
| [H128](../cuda/m4_gemm/04h_bn128.cu) | G 上改 BN128，2 CTA/SM | 1139.9 | 当前 4096³ 最佳，较 AB +74.2% |
| [H256](../cuda/m4_gemm/04h_bn256.cu) | G 上改 BN256，1 CTA/SM | 957.7 | 比 H128 慢，保留研究版 |

[完整对照及 NCU](results/m4-epilogue-tiles-20260923T044929Z/OBSERVATIONS.md)：四版各四形状与三轮 4096³ 全部严格对拍 PASS。G 的 store sectors/request 从 32 降到 4；H128 full 报告计数异常，已保留并另做有效的限定指标补采。NCU 原件都已下载。

随后单独同卡比较 stage：H128 S3=1140.7、[I：BK64 S2](../cuda/m4_gemm/04i_bn128_s2.cu)=1067.2、[J：BK128 S2](../cuda/m4_gemm/04j_bk128_s2.cu)=791.3 TFLOPS，均通过规定对拍。[stage 对照及三份 NCU](results/m4-stage2-20260923T045644Z/OBSERVATIONS.md)确认 shared 限制分别为 2、3、1 CTA/SM。I/J 编译需显式 `STAGES=2`。阶段深度、驻留数和末轮分配一起变化，不能仅凭某个 occupancy/stall 指标推断收益。

源码前后差异：[AB→G](04ab_warp_persistent-to-04g_epilogue.patch)、[G→H128](04g_epilogue-to-04h_bn128.patch)、[G→H256](04g_epilogue-to-04h_bn256.patch)、[H128→I](04h_bn128-to-04i_bn128_s2.patch)、[I→J](04i_bn128_s2-to-04j_bk128_s2.patch)。这次 epilogue 优化没有融合额外算子。

基线为用户的 [`03_pipeline.cu`](../cuda/m4_gemm/03_pipeline.cu)，SHA256 `fbec760bebdf20ee005c922d304233aad3e5e0824ac876806c15687588c0b599`。不在基线文件上试错；每种方法单独保存 `.cu` 和当次上传源码快照。首选判据是 B300 上 `4096³` BF16 GEMM 与基线**同一次分配交替三对**的中位吞吐，并要求候选在小形状和 `256×4096×16384` 严格对拍通过。未锁频的小幅差异不视为确定收益。所有程序均设单进程 5 秒超时、2 秒退出宽限，无自动重试。

| 版本 | 相对起点的单项变化 | 验证与决定 |
|---|---|
| [`04a_warp_specialized.cu`](../cuda/m4_gemm/04a_warp_specialized.cu) | TMA 生产者和 MMA 消费者分到不同 warp 的 lane 0；保留 S=3 tile、TMEM 及输出路径 | [四形状 PASS、同卡三对中位数 474.2→562.0 TFLOPS，+18.5%](results/m4-44-04a_warp_specialized-20260923T040637Z/OBSERVATIONS.md)：保留 |
| [`04b_persistent.cu`](../cuda/m4_gemm/04b_persistent.cu) | 独立从 4.3 派生；CTA 循环处理多个输出 tile，TMEM allocation 跨 tile 复用 | [四形状 PASS、同卡三对中位数 474.0→517.5 TFLOPS，+9.2%](results/m4-44-04b_persistent-20260923T041027Z/OBSERVATIONS.md)：保留 |
| [`04ab_warp_persistent.cu`](../cuda/m4_gemm/04ab_warp_persistent.cu) | 在保留的 warp 分工版上叠加 persistent 输出 tile 循环 | [与 04a 三对 555.3→654.1 TFLOPS，+17.8%](results/m4-44-04ab_warp_persistent-20260923T041351Z/OBSERVATIONS.md)；[与原 4.3 直接三对 472.0→651.6，+38.1%](results/m4-44-04ab_warp_persistent-20260923T042936Z/OBSERVATIONS.md)：当时最佳，现已被 H 超过 |
| [`04c_cta_pair.cu`](../cuda/m4_gemm/04c_cta_pair.cu) | 独立从 4.3 派生；两 CTA 合算 M=256、每 CTA B 行减半；CTA 0 单线程发起 cooperative MMA | [四形状 PASS，476.2→280.5 TFLOPS，−41.1%](results/m4-44-04c_cta_pair-20260923T041512Z/OBSERVATIONS.md)：只保留研究版 |
| [`04c_cta_pair_s4.cu`](../cuda/m4_gemm/04c_cta_pair_s4.cu) | 在 2-CTA S=3 版上将 stage 加深到 S=4 | [编译成功，最小用例 5 秒超时](results/m4-44-04c_cta_pair_s4-20260923T041645Z/OBSERVATIONS.md)：未验证 |
| [`04d_two_consumer.cu`](../cuda/m4_gemm/04d_two_consumer.cu) | CTA 0 两个 warp 各发一次不同 A 行的 group2 MMA，B tile 共享，输出 M 扩为 512；S=2 | [四形状 PASS，2-CTA 单→双 consumer 280.2→384.2，+37.1%](results/m4-44-04d_two_consumer-20260923T042141Z/OBSERVATIONS.md)：仍低于单 CTA 基线，保留研究版 |
| [`04e_two_consumer_remote.cu`](../cuda/m4_gemm/04e_two_consumer_remote.cu) | 两侧 TMA 直接报告 CTA 0 的 barrier，试图去掉逐 K cluster sync | [最小用例 5 秒超时](results/m4-44-04e_two_consumer_remote-20260923T042400Z/OBSERVATIONS.md)：未验证 |
| [`04f_two_consumer_handoff.cu`](../cuda/m4_gemm/04f_two_consumer_handoff.cu) | 各 CTA 等本地 TMA 后向 CTA 0 ready barrier 报到，两个 issue warp 分别等待 | [K=64/128 PASS，K=256 FAIL](results/m4-44-04f_two_consumer_handoff-20260923T042657Z/OBSERVATIONS.md)：未验证；[诊断源码](../cuda/m4_gemm/04f_two_consumer_handoff_diag.cu)也留存 |

**Epilogue 融合的适用性。**本题只要求 `D=A×B`，没有后续 bias、激活、残差或量化算子。直接在 epilogue 添加虚构运算会改变任务，不是可比较的融合实验。若只调整 `tcgen05.ld` 和 store 布局，那叫 epilogue **实现优化**，并非融合；目前先不把它冒充第四种方法。

仅在单项实测有效后另存组合源码，与被叠加的当前最佳版同卡比较。保留失败版本和完整原始记录，不因性能回退而覆盖源码。双 consumer 的含义是两个不同的 cooperative MMA，分别计算不同 A 行、共享 B，而**每一条** group2 MMA 仍仅由 CTA pair 中一个线程发起；参见 [PTX issue granularity](https://docs.nvidia.com/cuda/parallel-thread-execution/) 与 [MLC 第 9 步](https://mlc.ai/modern-gpu-programming-for-mlsys/zh/chapter_gemm_advanced/index.html#multi-consumer-warp-specialization)。

[两份本地 NCU 原生报告与分析](results/m4-44-ncu-20260923T041802Z/OBSERVATIONS.md)用于解释成功版与 2-CTA 初版的瓶颈；NCU replay 时间没有参与普通吞吐比较。本轮所有对拍仅覆盖表内形状，频率未锁定，不能外推到所有 GEMM 配置。
