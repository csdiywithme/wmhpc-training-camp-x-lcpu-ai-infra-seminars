# C1 / C2 最终交付

完成日期：2026-09-13。包含两题实现、B300 实验、详尽理论与测量报告、最终 writeup、10分钟答辩材料和内部独立审查。按用户确认，课堂环节交付准备材料和内部审查记录，未举行真实外组讨论或现场答辩。要求逐项核对见 [EXECUTION_STATUS.md](EXECUTION_STATUS.md)。

## 先读这两份 writeup

|题目|最终报告|核心结果|
|---|---|---|
|C1：FlashKDA / SM100|[C1 WRITEUP](c1_flashkda/WRITEUP.md)|原版相对同门函数FLA为1.48–2.89×；split2正确但六形状均变慢；保留当前默认，研究条件SM100后端|
|C2：MSA小batch decode|[C2 WRITEUP](c2_msa_decode/WRITEUP.md)|merge改良通过冻结验收，PDL关闭时1.066–1.268×；开启时有退化；CUTLASS交叉点依TP与成本口径变化|

这些是单卡 B300、固定源码与软件环境下的结果。TP 数字表示每卡 head 分片；没有运行多卡通信。C1微基准的局部收益、C2纯搬运收益及CUTLASS不同精度路径均未被混同为同精度完整算子加速。

2026-09-14补充了[面试准备总指南](INTERVIEW_GUIDE.md)、[C1连环追问](c1_flashkda/INTERVIEW_QA.md)和[C2连环追问](c2_msa_decode/INTERVIEW_QA.md)。材料依据上述实验，包含可口述答案、效率估算与工程白板题、反例及证据索引；非效率数学只保留实现所需的程度，供算子及推理框架岗位准备使用。

## C1：代码、完整分析和原始证据

|类别|入口|
|---|---|
|理论|[C1_THEORETICAL_ANALYSIS.md](c1_flashkda/analysis/C1_THEORETICAL_ANALYSIS.md) / [CPU检查](c1_flashkda/analysis/theory_checks.py)|
|GPU总报告|[C1_EXPERIMENTS.md](c1_flashkda/analysis/C1_EXPERIMENTS.md)|
|数值机制报告|[C1_NUMERIC_EXPERIMENTS.md](c1_flashkda/analysis/C1_NUMERIC_EXPERIMENTS.md)|
|挑战实现|[challenge_build.py](c1_flashkda/challenge_build.py)，在复制源码树中生成split2模块|
|tcgen05微基准|[tile_microbench.cu](c1_flashkda/tile_microbench.cu)|
|数值微基准|[chunk_numeric_gpu.py](c1_flashkda/chunk_numeric_gpu.py)，原实验快照差异在报告中说明|
|统一运行入口|[modal_experiments.py](c1_flashkda/modal_experiments.py) / [run_experiments.py](c1_flashkda/run_experiments.py)|
|答辩与审查|[DEFENSE.md](c1_flashkda/DEFENSE.md) / [FINAL_REVIEW.md](c1_flashkda/analysis/FINAL_REVIEW.md)|

最终证据目录：

- 官方形状：[H96](c1_flashkda/results/c1-official-h96-20260913T020630921340Z/) / [H64](c1_flashkda/results/c1-official-h64-20260913T022812279519Z/)。保留原样与同bounded-gate派生脚本。
- 原版计数器：[H96 NCU/SASS](c1_flashkda/results/c1-profile-h96-20260913T020442527901Z/) / [H12 NCU](c1_flashkda/results/c1-profile-h12-20260913T020932999579Z/)。
- 挑战：[正确性与性能](c1_flashkda/results/c1-challenge-h96-20260913T020956599379Z/) / [split2 NCU与完整形状对拍](c1_flashkda/results/c1-challenge_profile-h96-20260913T021340887396Z/)。
- 数值：[C16/32/64机制](c1_flashkda/results/c1-numeric-h96-20260913T021503565747Z/) / [最长32768弱衰减](c1_flashkda/results/c1-long_precision-h4-20260913T022617825299Z/)。
- tcgen05：[最终通过全CTA校验的tile实验](c1_flashkda/results/c1-tile-h96-20260913T022453478641Z/)。仅此最终方法用于writeup的tile结论，早期失败保留作历史。

## C2：代码、完整分析和原始证据

|类别|入口|
|---|---|
|理论|[C2_THEORETICAL_ANALYSIS.md](c2_msa_decode/analysis/C2_THEORETICAL_ANALYSIS.md) / [CPU检查](c2_msa_decode/analysis/theory_checks.py)|
|基线测量|[BASELINE_PROFILE.md](c2_msa_decode/experiments/BASELINE_PROFILE.md)|
|TMA实验|[TMA_EXPERIMENT.md](c2_msa_decode/experiments/TMA_EXPERIMENT.md) / [paged_copy.cu](c2_msa_decode/experiments/paged_copy.cu)|
|先测后设计|[CANDIDATE_DESIGN.md](c2_msa_decode/experiments/CANDIDATE_DESIGN.md)|
|最终候选|[candidate.py](c2_msa_decode/candidate.py)，`run(case)` 默认原partial+新merge|
|候选结果全表|[CANDIDATE_EXPERIMENTS.md](c2_msa_decode/experiments/CANDIDATE_EXPERIMENTS.md)|
|CUTLASS对照|[CUTLASS_EXPERIMENTS.md](c2_msa_decode/experiments/CUTLASS_EXPERIMENTS.md)|
|冻结验收|[ACCEPTANCE.md](c2_msa_decode/validation/ACCEPTANCE.md) / [suite.py](c2_msa_decode/validation/suite.py) / [CLI](c2_msa_decode/validation/run_validation.py)|
|运行入口|[候选](c2_msa_decode/modal_candidate.py) / [基线](c2_msa_decode/experiments/modal_baseline.py) / [TMA](c2_msa_decode/modal_copy.py) / [CUTLASS](c2_msa_decode/experiments/modal_cutlass.py)|
|答辩与审查|[DEFENSE.md](c2_msa_decode/DEFENSE.md) / [事前审查](c2_msa_decode/validation/INTERNAL_REVIEW.md) / [最终审查](c2_msa_decode/FINAL_REVIEW.md)|

最终数值和性能证据：

- [冻结校准168条](c2_msa_decode/results/candidate-calibrate-20260913T020759Z/calibration.json)，协议digest `430cf4da832b3fb3d2e3eeb2bff0c31959101a05703cc2ede89a1e10c4c54b02`。
- [最终heldout baseline168+candidate168，PASS](c2_msa_decode/results/candidate-verify-20260913T022150Z/candidate-heldout.json)。
- [最终同卡paired64记录](c2_msa_decode/results/candidate-paired-20260913T022747Z/paired.json)，含随机测量顺序、1344对样本与128份实际graph输出检查。
- [候选NCU](c2_msa_decode/results/candidate-profile-20260913T022426Z/)，新merge两个代表点无额外shared wavefront。
- [原基线矩阵](c2_msa_decode/experiments/results/bench-b300-20260913T015939Z/) / [初始NCU](c2_msa_decode/experiments/results/profile-b300-20260913T015957Z/) / [六个额外NCU切面](c2_msa_decode/experiments/results/profile-matrix-b300-20260913T020433Z/)。
- [TMA32配置](c2_msa_decode/results/paged-copy-20260913T015626Z/)。
- CUTLASS：[大batch](c2_msa_decode/experiments/results/cutlass-b300-20260913T020942Z/) / [小batch与FP8-P控制](c2_msa_decode/experiments/results/cutlass-b300-20260913T022105Z/)。

最终候选SHA256：`3e4a2ff87b352c7398ce814cf7ea81cf4e4c1fea33fbb5fe6bec8ee115644773`。当前源码与最终verify/paired/profile上传快照一致。

可导出图：

- 基线延迟：[PNG](c2_msa_decode/figures/baseline_latency.png) / [SVG](c2_msa_decode/figures/baseline_latency.svg)。
- TMA搬运：[PNG](c2_msa_decode/figures/paged_tma_copy.png) / [SVG](c2_msa_decode/figures/paged_tma_copy.svg)。
- 最终候选与PDL退化：[PNG](c2_msa_decode/figures/candidate_speedup.png) / [SVG](c2_msa_decode/figures/candidate_speedup.svg)。

## 复现和使用

两份 WRITEUP 末尾有从仓库根目录执行的命令。只阅读报告、运行CPU理论检查和重建已保存结果汇总无需GPU。重新运行Modal实验会产生平台费用，需要已有凭据、网络及相应GPU额度；本次授权运行均已结束。

不要把历史理论报告中的“尚待测量”或冻结协议中的“写入时尚未运行”误读为当前状态。它们是实验顺序与协议未篡改的记录；最终完成状态由本索引、两份WRITEUP、结果JSON和审查记录共同说明。未修改vendored上游，也没有将候选提交到外部仓库或部署为生产替代。
