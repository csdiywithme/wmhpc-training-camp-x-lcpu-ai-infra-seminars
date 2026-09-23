# C2 最终独立审查记录

日期：2026-09-13。审查角色：承担基线/CUTLASS 实验与独立性能核查的代理；最终候选代码、冻结验收方案和主 writeup 由其他代理完成。本记录明确区分源码检查、原始 GPU 证据、推断与未覆盖领域，不把代理互审称为实际外组意见。

## 当前交付范围

用户已确认以答辩材料和内部独立审查记录交付，真实外组审查与现场答辩在本次工作中由这些材料替代。实际课堂交流、现场提问及答辩没有举行，也没有向其他学员发送消息。这是用户同意的交付范围调整，不是对历史活动的补写。

按该范围核查，C2 的六个讨论点、挑战 a 的代码与 GPU 证据、可排练答辩材料均已具备。冻结协议不因状态更新而修改。

**审查结论：程序、原始证据和最终 [WRITEUP.md](WRITEUP.md) 已逐项核查。下述必要文案与复现命令修正已经完成；在用户确认的交付范围内，没有发现阻止 C2 交付的未解决技术问题。此结论不扩展为生产部署资格。**

## TASK 六个讨论点的证据核查

|讨论点|核查结论|直接证据|保留的边界|
|---|---|---|---|
|1．AI / Tensor Core|通过。使用 GQA16 的 KV 复用推导 `F=4RHqLD`、`Bkv=2RHkvLDs`，BF16/FP8 的 KV-only AI 为 16/32；计入 workspace 后下降。实际基线是 BF16 HMMA。|[理论推导及算术检查](analysis/C2_THEORETICAL_ANALYSIS.md)、[theory_checks](analysis/theory_checks.py)、[基线报告与 SASS 索引](experiments/BASELINE_PROFILE.md)|逻辑字节不等于 DRAM counter；不能拿 native FP8 Tensor Core 峰值解释反量化后的 BF16 dot。|
|2．融合 / cluster / mbarrier|通过可行性分析与已选路线证据。保留 split 并行，实际候选仅改 merge；两个代表点的 excessive shared wavefront 消除。cluster 的结合性、portable 8 CTA、共驻留和 DSM 生命周期均说明。|[理论第 8/9 节](analysis/C2_THEORETICAL_ANALYSIS.md)、[候选设计过程](experiments/CANDIDATE_DESIGN.md)、[候选 NCU](results/candidate-profile-20260913T022426Z/selected.csv)|没有实现 cluster，也没有把它当已测加速；普通 grid 的 mbarrier 不能创造任意 CTA 共驻留。|
|3．双级页表 / TMA|通过。线程做 top-k→block table 追踪，物理页确定后 TMA 做规则页内搬运；32 配置逐字节正确，实际 SASS 有普通 LDG 与 UTMALDG.4D。|[TMA 实验](experiments/TMA_EXPERIMENT.md)、[CUDA 实现](experiments/paged_copy.cu)、[原始数据](results/paged-copy-20260913T015626Z/result.json)|仅 copy 微实验，不包含 attention/反量化/流水线；不证明快于最优 cp.async；不能代替无效索引保护和 causal mask。|
|4．FP8 scale|通过。gold 按 baseline 的 represented-input 舍入；scalar 与变动的物理 token/head scale 均在 full GPU 验收域内，包含非 2 幂与 strided poison。另有 CUTLASS P-FP8 控制。|[冻结协议](validation/ACCEPTANCE.md)、[full heldout](results/candidate-verify-20260913T022150Z/candidate-heldout.json)、[P 舍入控制](experiments/results/cutlass-b300-20260913T022105Z/probability-controls.json)|默认 gold 不是未量化模型输入；移 scale 的实数等价不保证保留有限精度舍入；外部 CUTLASS 约 2.6% 误差没有被包装为冻结验收通过。|
|5．先 profile，再设计|通过。全 16 形状 baseline、首个成功 NCU 先于候选 GPU 探索；后续补齐 7 代表输入的 partial/merge NCU。小 B underfill、FP8 转换与部分 spilling、merge shared excess 分别有证据。|[完整基线](experiments/results/bench-b300-20260913T015939Z/measurements.json)、[首个 NCU](experiments/results/profile-b300-20260913T015957Z/baseline.csv)、[6 额外 NCU](experiments/results/profile-matrix-b300-20260913T020433Z)、[设计记录](experiments/CANDIDATE_DESIGN.md)|热 graph 与 cold-flush NCU 口径不同；少量 PC samples 不适合声称精确瓶颈占比；历史 NCU 失败不代表本轮没有计数器。|
|6．独立验收 / 先固定方案|通过当前确认的替代交付范围。独立 FP64 gold、baseline-only calibration、固定 seeds/floor/cap、哈希冻结、独立 heldout 和内部挑战记录齐全；full 有 baseline168+candidate168 PASS。|[验收协议](validation/ACCEPTANCE.md)、[内部审查](validation/INTERNAL_REVIEW.md)、[冻结 manifest](results/candidate-verify-20260913T022150Z/calibration.json)、[full 结果](results/candidate-verify-20260913T022150Z/candidate-heldout.json)|真实外组意见未发生；按用户同意由内部独立审查记录替代。PASS 只覆盖冻结域，不等于任意生产 Graph/PDL、共享 prefix、sanitizer 或模型质量证明。|

## 挑战 a 与性能方法核查

- 最终 [candidate.py](candidate.py) 保持 `_page_decode_kernel` 与原 `_gqa_sparse_decode_kernel` 重命名后的 AST 完全相同。选中路线为 feature-tiled merge、1 warp；S≤8 使用 Dtile128，否则 Dtile64。历史 subpage/tile/stage 探索保留为探索，不冒充最终启用的路径。
- [最终 paired](results/candidate-paired-20260913T022747Z/paired.json) 有 16 个 TP/batch/dtype 形状 × 2 个 heldout seed × 2 个 PDL 状态，共 64 条记录；每条 21 对随机交错样本，共 1,344 对。两端相同输入、PDL、设备与预分配策略，每张 graph 64 次调用。
- `samples[baseline]` 与 `samples[candidate]` 每个 repeat 各追加一次，虽随机交换执行顺序，按索引计算配对 ratio 仍正确。ratio 的无量纲字段已与微秒统计区分。
- replay 后检查两端共 128 份输出；全部有限，最大全局 NRMSE=0.0029301754，最大绝对误差=0.0002346012。此处是性能形状的补充 sanity check，不取代独立冻结 full suite。
- PDL=false 的全部 32 条 seed 记录 speedup 范围 1.0661608–1.2683512；不能把它误标成 32 个不同形状。PDL=true 在 TP1 B1 和 TP4 B4 的 BF16/FP8 均有退化，不能给出无条件生产替换结论。
- 两个 candidate NCU 代表点确实有 `.ncu-rep`/CSV：TP1 B1 merge wavefront actual/ideal=640/640，TP1 B16=0/0；不是只从共享内存分配量推断“bank conflict 已消除”。对应 dynamic shared 为 128B/0B。

## CUTLASS 对照核查

固定的是 vLLM `d4da0c55af3aa231b6209bf77871f3ed36eab0d2` 实际依赖的 MSA `087c161814d4d9c735b46c21212a09e5f8eb92fa`，CUTLASS submodule `eb61c911471867a5fd2466bfd8f29306cea6ebf8`。保留公开归档哈希、编译源码、flags、.so、resource、SASS 与 CUDA activity；没有调用安装环境里的任意最新版实现。

[CUTLASS_EXPERIMENTS.md](experiments/CUTLASS_EXPERIMENTS.md) 的每一性能行均为同卡同任务对照。B1/4/8/16 与 B32/64 是两次任务，报告显式区分；B<16 强制调用底层 wrapper 只作实验，没有伪装成生产 guard 支持。

所测离散点：TP1 attention-only 在 B8 开始获益，本实验含 Q 量化与热 metadata 的 full graph 在 B16 获益；TP4 对应 B32/B64。没有复现 TP4 cross16，也没有测尽所有中间整数 batch 来确定精确交点。full graph 不含 Python/冷 plan/服务调度，其普通 PyTorch Q 量化也不冒充生产融合量化成本。

随机重复页控制的数学 gold NRMSE 为 0.02606014；加入独立 P-FP8 numerator 舍入后降到 0.00168129，Q=0 时为 0.00166285。与固定源码 `NumericArrayConverter<Element,...>` 和 SASS FP8 pack 一致，支持 P 舍入解释主要额外误差。该诊断是额外证据，不修改任何 frozen gate，也不声称穷尽全部误差来源。

## 文件一致性与可复现性核查

核查时，当前 `validation/suite.py`、`validation/ACCEPTANCE.md`、`validation/run_validation.py`、`vllm_msa_ref/sparse_attn.py`、`harness/vllm_shim.py` 的 SHA256 均与 full calibration manifest 记录一致。更新本审查与交付范围没有触碰这五个文件。

当前 candidate SHA256 为 `3e4a2ff87b352c7398ce814cf7ea81cf4e4c1fea33fbb5fe6bec8ee115644773`，与最终 `candidate-verify-20260913T022150Z`、`candidate-paired-20260913T022747Z`、`candidate-profile-20260913T022426Z` 三份源码快照完全一致。数值、性能与 NCU 使用同一份最终候选。

基线/CUTLASS 派生报告可由 [summarize_baseline.py](experiments/summarize_baseline.py) 与 [summarize_cutlass.py](experiments/summarize_cutlass.py) 从原始 JSON/CSV 重建；[DEFENSE.md](DEFENSE.md) 提供讲稿、问答及可打开的证据索引，其本地链接已经检查存在。

## 最终 WRITEUP 的修正与复核

1. **复现命令已修正。** `modal_cutlass.py` 的实际入口是 `main(batches="16,32,64", tps="1,4", pdl=True, controls=False)`，没有 `small`/`wide` 模式。主报告现在给出 `--batches 16,32,64` 与 `--batches 1,4,8,16 --controls` 的完整 Modal 命令，并补 baseline PDL 开关、两类 NCU 命令及 CPU CUTLASS 报告重建入口。命令参数与本地入口核对，本次审查没有重新启动付费 GPU。
2. **精度资格措辞已收紧。** CUTLASS 仅做指定性能形状和独立概率控制，没有运行完整冻结候选 gate；因此记录为“未取得该资格”，不冒称存在一份 full-suite FAIL 或 PASS。约 2.6% 的参考误差保留，验收阈值没有修改。
3. **空 split 的数学条件已补充。** 非空行使用 log2-LSE 质量权重；空 split 写局部输出 0、LSE 为负无穷，全 padding 行按接口忽略。避免将 `Z_c=0` 直接代入局部归一化公式。
4. **源码保存范围已精确化。** 完整上传文件快照属于候选任务；baseline/CUTLASS 保存运行器和 workload 快照及版本/环境/命令证据，TMA 保存源码哈希和命令输出。没有为历史任务追写原始数据或虚称每个任务均保存全部上传源码。
5. **公式与数字复核通过。** `4GLD=16,777,216` FLOP，KV-only AI=16/32 FLOP/B，加入一次 Q/O 后为15.876/31.508；split/CTA 表与源码一致。重新从最终 `paired.json` 聚合，主报告7行代表时间、PDL=false 的1.0661608–1.2683512及几何平均1.1174897、PDL=true 的4组退化均一致。128份 graph 后输出检查存在，最大全局 NRMSE=0.0029301754；与168条 heldout 使用不同误差口径，正文已区分。
6. **官方文档链接已打开核实。** [CUDA 编程模型与 cluster](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#thread-block-clusters)、[PTX mbarrier](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier)、[Driver tensor map API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)、[PTX tensor copy](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor) 均指向 NVIDIA 官方文档。所引功能不被夸大成任意全局 CTA 汇合或自动追踪两级任意索引表。
7. **结论边界复核通过。** TP1 attention/full 的离散首个获益点为B8/B16，TP4为B32/B64；小/大矩阵不是单一同卡曲线。TMA仅copy；最终候选收益限定PDL=false课程域，PDL退化成因仍未确证；full frozen验收不冒充生产PDL/服务/共享prefix/sanitizer验收。主报告、答辩稿和内部审查均明确用户同意的课堂材料替代范围。

最终检查 `WRITEUP.md`、`DEFENSE.md`、`FINAL_REVIEW.md` 和 `validation/INTERNAL_REVIEW.md` 的本地 Markdown 链接均存在；原始 JSON、NCU 报告和冻结文件未因本次文档审查而修改。

## 历史 pending 描述如何理解

- 冻结 `ACCEPTANCE.md` 中“写入时尚未运行 GPU”“仅计划内部审查”是协议写定时点，保留原文保证 digest；当前状态由 `INTERNAL_REVIEW.md` 的追加说明和原始 full JSON 证明。
- `CANDIDATE_DESIGN.md` 的“完整矩阵仍在执行”“下一项实验”等是探索期间的设计记录，不是当前最终状态；不要删除历史失败/未完成时点来制作只含赢家的叙述。
- 先前理论报告中的“待 profile/待实测”是理论阶段限制；本轮应以 `BASELINE_PROFILE.md`、`TMA_EXPERIMENT.md`、`CUTLASS_EXPERIMENTS.md` 和最终 writeup 接续它，而不将理论阶段推断追写成当时已有 GPU 证据。
- 根代理已将总体 `EXECUTION_STATUS.md` 和交付索引同步为最终证据状态；真实课堂活动以用户确认的材料范围替代交付，未改写为活动已举行。
