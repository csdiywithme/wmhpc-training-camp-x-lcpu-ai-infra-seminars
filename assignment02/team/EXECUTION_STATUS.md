# C1 / C2 要求与交付核对

更新：2026-09-13。目标为两题的复现、六个讨论点、挑战实现、详细报告、writeup 与答辩材料。入口见 [DELIVERABLES.md](DELIVERABLES.md)。本表取代实验开始时的待办状态；历史理论、设计记录和冻结验收协议保留写定时的文字。

**两题技术工作均已完成。课堂环节按用户明确确认，以答辩材料与内部独立审查记录交付；未举行真实外组挑战或现场答辩。** 所有本轮 GPU 工作已结束，没有需要继续运行的实验。

## 执行边界

- 用户明确授权上传必要源码并运行付费 Modal B300/B200。本次使用单卡 B300；不宣称 B200 实测或多卡 TP 通信。
- 各运行器设超时且无自动重试；保存命令、退出码、环境、原始样本，并按各运行器保留源码快照或哈希。没有留下持续执行或周期性 GPU 作业。
- C1 `FlashKDA/`、`fla_kda_ref/` 和 C2 `vllm_msa_ref/` 上游快照保持不变。候选及实验在独立文件中实现，未改动无关练习。
- C2 先 baseline profile，再候选设计；验收由 baseline 校准后冻结，最终候选未用于生成或放宽阈值。
- 保留失败和负收益。静态 SASS、NCU counter、CUDA Graph 计时、CPU 反例和纸面外推分别标明，不相互冒充。
- NCU 历史失败已在本轮解决；当前结论使用实际成功的 detailed profile，不再将旧 probe 失败列为交付阻塞。

## C1 要求与完成证据

|要求|状态|证据|
|---|---|---|
|B300 安装 pinned FlashKDA|完成|`1ce47ea` / CUTLASS `5c149f5`，sm_103a 实际构建，环境与命令保存|
|官方 benchmark 复现|完成|H96/H64 × 固定8192 / 不等长6序列 / 8×1024；原样及同 bounded gate 比较均保存|
|SM80 主路径确认|完成|实际扩展 SASS 为 HMMA BF16 / F16；原版未出现 tcgen05 矩阵指令|
|1：C16 数值 / 求逆 / tile|完成|C16/32/64 理论成本、6指数案例、45求逆案例；随机与结构性失稳反例|
|2：tcgen05 形状与微基准|完成|转置形状推导；最终64配置全CTA有限/gold检查，320计时样本；旧失效方法不用于结论|
|3：递推并行度与挑战|完成|候选方案逐项反例；split2 CUDA实现、30精度案例、8接口组合、H96三完整形状逐位通过；六计时形状负收益|
|4：真实瓶颈|完成|H96/H12 K1/K2 detailed NCU、SASS、资源和state-chain切面对照；split2另有profile|
|5：BF16 状态精度|完成|官方舍入oracle、独立FLA naive FP32、FLA chunk；30普通案例和4个最长32768弱衰减案例、1024-token窗口|
|6：v2 决策|完成|保留现有默认、研究条件后端，不发布当前split2；精度、并行度、端到端收益与移植边界均明确|
|writeup / 报告 / 答辩 / 审查|完成|[WRITEUP](c1_flashkda/WRITEUP.md)、[实验报告](c1_flashkda/analysis/C1_EXPERIMENTS.md)、[数值报告](c1_flashkda/analysis/C1_NUMERIC_EXPERIMENTS.md)、[DEFENSE](c1_flashkda/DEFENSE.md)、[FINAL_REVIEW](c1_flashkda/analysis/FINAL_REVIEW.md)|

C1 的负收益挑战符合 TASK 的完成条件。没有把“本次 split2 不值得采用”外推为“所有 SM100 实现不可能获益”。完整 tcgen05 K2、大 chunk KDA、模型级质量与 FP32 持久状态消融未实现，均作为已声明的后续研究范围。

## C2 要求与完成证据

|要求|状态|证据|
|---|---|---|
|B1/4/8/16 baseline profile|完成|TP1/4×BF16/FP8，16形状×PDL开关；partial/merge/chain分解；7个代表输入成功NCU|
|先测量后设计|完成|baseline `015939Z`、首NCU `015957Z`，后续设计与候选 `021041Z` 等时间戳及来源记录|
|1：算术强度 / Tensor Core|完成|GQA16推导16/32 FLOP/B（KV理想口径）、workspace/scale限制；HMMA实际指令和grid/利用率|
|2：融合 / cluster|完成|LSE归约代数、split CTA表、cluster/mbarrier约束；实现保留两kernel并实测PDL正负结果，未冒称已实现cluster attention|
|3：间接寻址 / TMA|完成|文档及CUDA微实验：SM两次LDG追表+UTMALDG.4D；32配置逐字节gold通过|
|4：FP8 scale|完成|物理token/head索引与数学/舍入层分析；scalar非2幂、真实变化scale、stride和页重排均验收|
|5：实际瓶颈|完成|冷NCU与hot graph分开；小batch并行不足、FP8转换/布局/部分spill、merge excess wavefront证据|
|6：自定验收|完成|独立FP64 gold；baseline168校准冻结；heldout baseline168+最终candidate168均PASS；事前内部审查记录|
|挑战(a)：改良 Triton|完成|原partial+单warp feature-tiled merge；独立seed paired64记录、1344对样本及128份graph后输出；PDLfalse全部获益，PDLtrue退化保留|
|CUTLASS B16+ 对照|完成|固定外部MSA/CUTLASS实际构建；B1/4/8/16/32/64同卡逐行对照；量化/metadata/attention成本与FP8-P数值诊断|
|writeup / 报告 / 答辩 / 审查|完成|[WRITEUP](c2_msa_decode/WRITEUP.md)、[候选报告](c2_msa_decode/experiments/CANDIDATE_EXPERIMENTS.md)、[CUTLASS报告](c2_msa_decode/experiments/CUTLASS_EXPERIMENTS.md)、[DEFENSE](c2_msa_decode/DEFENSE.md)、[INTERNAL_REVIEW](c2_msa_decode/validation/INTERNAL_REVIEW.md)、[FINAL_REVIEW](c2_msa_decode/FINAL_REVIEW.md)|

C2 的 frozen gate 未因 CUTLASS 而放宽，CUTLASS 外部对照没有被登记为已通过候选 gate。真实服务延迟、共享prefix、sanitizer及全部边界的生产PDL路径不是本次PASS的隐含覆盖；当前候选不作无条件生产替换。

## 最终核对口径

源码哈希与成功 GPU 快照一致；冻结协议文件与manifest一致；本地引用和最终数表均交叉检查。核对只重读已保存产物与必要的本地检查，没有因为整理报告而再运行收费 GPU 实验。性能可复验但不保证在未锁频的另一张卡上逐微秒复现；数值 PASS 不等于所有输入、其他架构或模型质量的形式保证。
