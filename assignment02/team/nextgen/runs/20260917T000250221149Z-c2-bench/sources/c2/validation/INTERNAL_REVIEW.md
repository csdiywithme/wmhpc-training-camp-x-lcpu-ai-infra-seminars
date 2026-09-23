# C2 验收方案独立审查

2026-09-13，主代理在候选 GPU 运行前审查 `ACCEPTANCE.md`、`suite.py` 与 `run_validation.py`。方案由独立代理编写。本记录只证明内部审查，不冒充与其他学员小组的讨论。

## 当前交付范围与状态说明

用户已确认，本次按答辩材料和内部独立审查记录交付即可，真实外组审查与现场课堂活动以这些材料替代。本记录没有把内部代理互审升级为真实外组意见；实际外组交流、现场提问和答辩仍未举行，也未向其他学员发送消息。

`ACCEPTANCE.md`、`suite.py`、`run_validation.py` 是 manifest 锁定的协议内容，未为了更新交付状态而编辑。协议末尾“本协议写入时尚未运行 GPU 验收”和“仅计划……审查”等内容描述冻结前时点，不表示现在尚无结果。当前证据是：

- [full calibration](../results/candidate-verify-20260913T022150Z/calibration.json) 状态为 `CALIBRATION_FROZEN`；[独立 heldout](../results/candidate-verify-20260913T022150Z/candidate-heldout.json) 有 168 条 baseline 与 168 条 candidate，整体 `PASS`。
- [最终成对性能记录](../results/candidate-paired-20260913T022747Z/paired.json) 包含 64 个输入/PDL/seed 组合、每组 21 对随机交错样本；replay 后共检查 128 份输出，全部有限，最大全局 NRMSE 为 0.00293018。该补充检查不等于将冻结 full suite 扩展为全部生产 Graph/PDL 输入。
- [答辩材料](../DEFENSE.md) 已包含 10 分钟讲稿、5 分钟主问答及备用问题，供用户排练与提交；没有制作虚构的现场问答纪要。

下文审查问题保留原时序，后续补充不改变阈值、gold 定义或历史原始结果。

## 审查问题与处理

1. **随机尾页测试是否能抓住 causal off-by-one？** 单个未来 token 对长序列均值贡献可能小于容差。方案已加入 `causal_probe`：Q=0，除最后 token 的 V=8 外其余 V=0；早 query 应输出严格零。此修改在 baseline calibration 与候选 GPU 测试之前写入协议。
2. **FP8 gold 是否比较同一份数值输入？** 默认 gold 显式执行 FP8→BF16、FP32 scale 相乘、BF16 舍入，再以 FP64 attention 计算。另一个 raw-scale gold 被清楚标为诊断口径，不能用来偷换验收。
3. **常量 scale 是否掩盖地址错误？** token/head scale 随物理 token、page 和 head 改变；存在 stride=2 poison backing；物理重排同时变换 cache、page table 和 scale。
4. **空行 NaN 会不会导致误判或漏检？** `active_rows` 显式区分 padding。活跃非有限数是独立失败项，不能因为其他误差字段为零而通过。全空 gold 采用零存储仅为约定。
5. **是否用候选结果决定容差？** calibrate 模式禁止 candidate adapter，固定 calibration/heldout seeds，超过硬 cap 不生成冻结成功状态；manifest 包含协议/源码 hash，验证时拒绝不一致的协议。
6. **是否只跑少数案例就声称全部通过？** runner 按 spec 与变体计算期望记录数，异常提前停止时记录数不足，不能形成 PASS。quick 与 full 在输出和文档中分别标注。
7. **候选是否可复现？** Modal runner 另存本次上传的 candidate、workload 和 validation 源码快照；候选不能修改冻结的 baseline manifest。

## 仍然限定的范围

本轮域是 D128/page128/topk16/GQA16、TP1/TP4、BF16 Q 与 BF16/E4M3FN KV。没有把 PASS 扩展到共享 prefix、任意 head ratio、E5M2、训练反向或完整 vLLM 引擎。PDL/Graph 的性能与补充正确性需要单独记录。

上述 PDL/Graph 单独记录已由最终 paired 实验补上；它覆盖正常 seq8192 性能形状的 PDL 开/关及图后输出，不覆盖 frozen suite 全部边界在生产 graph 调度中的行为。PDL 开启时存在退化形状，因此内部审查没有批准“无条件生产替换”这一结论。

审查后开始原 baseline 的 full calibration。任何后续协议修改必须保留旧结果、重新版本化和校准，不能为候选放宽门槛。
