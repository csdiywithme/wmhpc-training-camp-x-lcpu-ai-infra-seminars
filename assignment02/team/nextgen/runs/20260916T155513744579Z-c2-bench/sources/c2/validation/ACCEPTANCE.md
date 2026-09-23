# C2 验收方案 v1：represented-input attention

日期：2026-09-13。本文是实现候选前写定的验收协议，适用于题目 vendored Triton baseline 与通过同一 adapter 接入的候选。它不是模型质量证明，也不是性能测量。外部小组讨论尚未进行；仅计划由主代理和其他代理独立审查，不能把内部审查写成已经完成题目要求的跨组挑战。

## 1. 支持域与输出契约

- D=128，page=128，topk=16，GQA=16；TP1 为 64Q/4KV，TP4 为 16Q/1KV。
- Q、output 与 partial 基线为 BF16；KV 为 BF16 或 E4M3FN FP8。其他 FP8 格式和 FP16 Q 不属于本轮冻结域。
- K/V scale 同时省略、同时为 scalar，或同时为同形的 FP32 `[KV_head, physical_token]` 表。scale 有限且为正；生成器保证 Q/K/V 与反量化结果可表示，通常幅度约 0.5，sharp 模式只增大 Q。
- `seq_lens` 包含本轮 query；flattened query 按 request-major 排列。普通 dql=1，显式测试 dql=2/4；非 padding 请求满足 `seq_len >= dql`。
- 每个活跃 top-k 前缀长度为 `min(16, ceil(max(seq_len-dql+local_q+1,0)/128))`，其中逻辑块必须有效且无重复。当前块由生成器选入。尾槽会使用不同负值作为 poison，kernel 必须按长度忽略，不能按 sentinel 终止。
- `block_table` 将逻辑块映射到打乱的物理页；scale 下标必须是 `physical_page*128+offset`。top-k 测试生产式 token-major backing 的转置 view，scale 表可为 stride=2 的 view，且 backing 间隔槽存负 poison。
- 活跃输出须全部有限，shape、dtype、device 与 Q 一致。全空请求只作显式 Graph padding；padding 输出允许 NaN 或零，不参与数值比较，但记录非有限数。它们不得影响相邻活跃请求。
- 非法活跃索引、错误 shape、非正/NaN scale 不直接传入 GPU 制造越界；这些需由上层契约或单独的拒绝测试处理。未选中/因果不可见数据的 NaN 污染测试只在合法已分配地址中进行。

生成器默认每请求使用不共享的物理页。真实共享 prefix、生产 Graph/PDL 路径、其他 dtype/head ratio 和 sanitizer 仍是补充验证项，不隐含在本轮 PASS 中。

## 2. 独立 gold 与数值口径

`suite.gold(case)` 不调用 SDPA、Triton baseline、候选、online softmax 或 split merge。它显式按请求、KV head、有效 top-k 前缀聚齐 K/V，用逻辑位置裁去未来 token，然后进行 FP64 QK、减最大值、自然指数归一化和 FP64 PV。结果保留 FP64，不预先舍入成 BF16 来缩小误差。

默认 FP8 gold 精确说明了输入舍入：先将 FP8 解码到 Q dtype，再以 FP32 scale 相乘、舍入回 Q dtype，最后升为 FP64 做 attention。这与当前 Triton 的有效 K/V 输入一致，比较的是矩阵乘/softmax/概率转换/partial/merge 的数值误差。FP8 输入值在本域内转换到 BF16 可精确表示。

`gold(case, raw_scale_gold=True)` 是另一种诊断口径：FP8 数值和 scale 直接在 FP64 相乘，不执行 BF16 反量化舍入。它不能与默认 gold 偷换，也不用于本轮冻结门槛。未量化 KV、原 BF16 Q 与 CUTLASS 额外的 FP8 Q 的差异属于另一层量化/模型质量测量。

全空 row 在 gold 中写零，并同时使用显式 active mask；这只是存储约定，不声称空集合 softmax 有数学定义。

## 3. 先 baseline 校准，再冻结，再候选验收

协议常数在 `suite.POLICY` 固定。校准 seeds 为 11/29，独立 heldout seeds 为 101/307。quick 是覆盖性检查；full 才覆盖全部列出的 head/dtype/batch 主矩阵。每个结果必须注明 tier，quick PASS 不能写成 full 验收通过。

baseline 校准不接收 candidate adapter。按 `storage:pattern` 族聚合所有 calibration 样例和等价变体的误差最大值 b，再一次性生成阈值：

| 指标 | 冻结阈值 | 独立硬上限 |
|---|---|---|
| max absolute error | min(0.02, max(0.002, 2b+0.0005)) | 0.02 |
| 每 query/head 的 NRMSE 最大值 | min(0.10, max(0.02, 2b)) | 0.10 |
| 全局相对 L2 | min(0.05, max(0.02, 2b)) | 0.05 |
| elementwise `abs(error)/(0.02+0.02*abs(gold))` | 1 | 1 |
| 活跃输出非有限数 | 0 | 0 |

每行 NRMSE 定义为 `error_RMS / max(reference_RMS, 0.01)`，另报告 P50/P95/P99/max。全局相对 L2 分母设数值下限 1e-12；接近零的输出仍同时受绝对误差和逐行检查约束。

这些常数是针对上述有限幅度输入域、在候选之前提出的回归政策，不是从 FP8 位数推出的定理。0.02 elementwise 门槛沿用上游测试作为硬保护；倍数、floor、cap 的组合给 baseline 留有限的未见样例裕量，避免一条随机样例的近零误差生成不切实际的零容差。逐行与绝对门槛防止全局 L2 隐藏局部错误。

若 baseline 本身违反任意硬上限、抛异常或没有跑完全部样例，则不生成成功冻结状态。若 heldout baseline 违反冻结阈值，candidate 不运行，结果是 `BASELINE_HOLDOUT_FAILED`，不能把它记作 candidate 失败或通过。需要分析原因时只可创建新协议版本、新路径、新校准，保留旧结果；不得看候选结果后持续放宽标准。

manifest 保存协议、全部 spec、seeds、baseline maxima、thresholds，以及 suite/CLI/本文件/vendored kernel/shim 的 SHA256。验证时重新计算 digest，不一致则拒绝复用阈值。文件用独占创建，禁止覆盖已有 calibration/result。候选结果另存，不写回 baseline manifest。

## 4. 覆盖矩阵和不变量

quick 覆盖 TP1/TP4、三类存储、短序列边界、dql 跨页、混合 padding、全空 padding、sharp logits，以及两种存储下的页映射/顺序/poison 不变量。另设 causal_probe：Q=0、所有 V=0、仅请求最后一个 token 的 V=8；早 query 必须输出0，最后 query 才可读到8，使未来 token 泄漏明显超过固定绝对误差上限。

full 在 quick 上增加每个 head/storage 组合的 batch 1/4/8/16、8192 长度、dql 2/4 跨 128/256/2048 边界、混合 padding、Q=0、常量有效 V、sharp logits 和正负消去。固定 spec 在 `build_specs()` 中可审查，未声称支持的 15/17 dispatch、dql=32 等需另加协议版本。

对标有 metamorphic 的输入执行：

1. 反转每个有效 top-k 前缀，保持集合不变。
2. 一致置换物理 KV 页、block table、物理 scale 表。
3. dql=1 时，将不被当前 head 选中的页与尾页未来 token 改为 NaN。dql>1 不用这项全 row 不变量，因为早 query 的未来 token 对晚 query 可能合法可见。

每个变体均独立计算 FP64 gold，先确认与原 gold 的最大差不超过 1e-10，再比较 kernel。变体与原 kernel 输出另按两个绝对误差界的裕量检查；允许 finite-precision 归约顺序改变，不要求 bitwise 相同。测试生成器具有 request-private pages；不要将 `variants()` 的污染逻辑直接用于任意共享物理页输入。

Q=0 样例的 gold 是所有选中可见 token 的 V 平均，不是各页平均再平均；常量 V 模式使反量化后的有效 V 恒为 0.75，避免把 prequant 常量误认成 postquant 常量。FP8 scalar 使用 0.37/0.63，token/head scale 真实变化且非 2 的幂，避免上游常数 scale 测试的盲点。

## 5. 运行和接入

从 C2 目录运行，GPU 环境需 Torch+Triton；无需安装完整 vLLM。baseline 通过原 harness shim 加载本题快照，明确关闭 PDL。

```sh
python validation/run_validation.py probe --tier quick --output results/validation-probe.json
python validation/run_validation.py calibrate --tier full --output results/validation-calibration-v1.json
python validation/run_validation.py verify --manifest results/validation-calibration-v1.json --output results/validation-baseline-heldout-v1.json
python validation/run_validation.py verify --manifest results/validation-calibration-v1.json --adapter my_candidate:run --output results/validation-candidate-heldout-v1.json
```

adapter 为 `run(case: dict) -> Tensor`，返回 Q shape/BF16/device 的输出；其实现不得读取 gold。可用 `module:function` 或 `/absolute/file.py:function`。性能实验可直接复用接口：

```python
from validation.suite import CaseSpec, make_case, baseline, gold, evaluate
spec = CaseSpec("example", kv_heads=4, seq_lens=(8192,), storage="fp8_token")
case = make_case(spec, seed=101, device="cuda")
reference = gold(case)
metrics = evaluate(case, baseline(case), reference)
```

正确性调用包含 FP64 gold、CPU metadata 读取和显式同步，不得把其耗时写成 kernel latency。性能计时应另行隔离 kernel/API、warmup、allocations、cache、PDL/Graph 和实际 backend。

## 6. 审查与完成状态

本协议写入时尚未运行 GPU 验收，静态检查不构成数值通过。后续结果以 JSON 的环境、协议 digest、实际状态和逐 case 记录为准。内部代理审查、GPU 校准、heldout、candidate 验收、生产 Graph/PDL/sanitizer、真实跨组讨论应分别记录，不能相互替代。
