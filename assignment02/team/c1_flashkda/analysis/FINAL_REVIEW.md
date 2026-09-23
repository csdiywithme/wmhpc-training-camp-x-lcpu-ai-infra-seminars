# C1 最终交付内部独立审查

审查日期：2026-09-13。对象为 [WRITEUP.md](../WRITEUP.md) 和 [DEFENSE.md](../DEFENSE.md)。本次由未撰写这两份初稿的协作审查者完成，依据既有源码和原始实验产物交叉核对；不构成另一台 GPU、另一软件环境或模型质量的独立复现实验。本轮没有运行 GPU、追加性能实验或修改上游快照。

按用户确认，课堂环节以答辩材料及本内部独立审查记录交付。**未实际举行课堂讲述、现场问答或答辩，不编造听众反馈、现场评分或课堂完成记录。** DEFENSE 中的 10 分钟讲稿、5 分钟问答和八页内容安排是可使用的准备材料，不是已发生的活动。

## 审查结论

两份材料覆盖 [TASK.md](../TASK.md) 的复现、六个讨论点和挑战实现。主要数字与 [C1_EXPERIMENTS.md](C1_EXPERIMENTS.md)、[C1_NUMERIC_EXPERIMENTS.md](C1_NUMERIC_EXPERIMENTS.md) 及抽查的原始 JSON 一致；必要措辞修正已直接写入终稿。没有发现需要撤回现有性能表、重新运行 GPU 或阻止本次交付的错误。

结论强度适当：保留当前默认路径有实测依据，但没有宣称传统 MMA 对所有 SM100 设计全局最优；split2 的负收益限定为当前具体实现；tile 增量优势没有被写成完整 KDA 加速；有限精度验证没有被写成全部输入或模型质量保证。

## 六个讨论点逐项核对

| TASK 讨论点 | 终稿位置 | 核对结果与适用范围 |
|---|---|---|
| 1：C16 的数值范围、Neumann 与形状匹配 | WRITEUP §4；DEFENSE 第 3 页、问答 1–3 | C16/32/64 的 dense 求逆次数 6/8/10、每 token FLOP 3,072/16,384/81,920 与推导一致；第 18 token 的指数归零/逆因子 Inf、C16 抵消及 C32/64 部分和溢出有机制实验。shared 和完整大 chunk 成本清楚标为纸面外推。 |
| 2：tcgen05 形状与仅换指令 | WRITEUP §5；DEFENSE 第 4 页、问答 4–5 | 区分 K1 M16 padding 与 K2 转置后合法形状；N 的支持范围已补全。只引用最后修正后的 tile 数据，说明固定 grid 增量、TMEM/同步及 SM80 寄存器复用。未宣称峰值或完整 K2 收益。 |
| 3：其他并行来源与挑战 | WRITEUP §6；DEFENSE 第 6 页、问答 6–8 | 真实独立序列、value split、多 head、persistent、两 CTA 与 affine scan 均给出候选及反例。split2 是真实 CUDA 并行度重构，正确但六个正式形状都变慢，符合 TASK 允许负收益挑战的范围。 |
| 4：瓶颈与 NCU 证据 | WRITEUP §7；DEFENSE 第 5 页、问答 10–11 | K1/K2 分开分析；H96/H12 的 grid、时间与 elapsed/active 指标形成交叉证据。保留 NCU replay/cache/频率条件和 benchmark 区别，不将 stall 采样当作精确耗时归因。 |
| 5：BF16 状态精度 | WRITEUP §8；DEFENSE 第 7 页、问答 9 | 三层参考、30 组小形状、4 组最长 T32768 弱衰减及窗口指标可追溯。naive 明确为 FP32；状态方向停滞例明确为 CPU 隔离反例；尚未做完整持久状态精度消融与模型级质量评估。 |
| 6：v2 与可移植性 | WRITEUP §9；DEFENSE 第 8 页 | 保留默认、继续研发条件后端、不发布当前 split2 的决策与数据一致。“SM80 MMA”没有被混同为整个 kernel 可运行在 Ampere；B300 实测目标为 sm_103a，其他架构不作实测保证。 |

## 数字与原始证据抽查

- 官方 benchmark：核对 H96/H64 各三种序列组织的原样及同 `safe_gate` 派生结果。终稿同语义速度比范围为 **1.48–2.89×**，H96 单独为 **1.93–2.89×**；原始与派生 benchmark 未混用。见 [H96](../results/c1-official-h96-20260913T020630921340Z/)、[H64](../results/c1-official-h64-20260913T022812279519Z/)。
- split2：六组 baseline/candidate 时间与 **0.571–0.751×** 范围一致；抽查 [precision.json](../results/c1-challenge-h96-20260913T020956599379Z/precision.json) 共 30 组，官方 oracle 与候选 output/state 的 `exact` 均为真。八种接口检查来自脚本中的 `torch.equal` 断言及成功退出，不误称为另有八份独立 JSON。完整 H96 三形状另有 [显式正确性记录](../results/c1-challenge_profile-h96-20260913T021340887396Z/large-shape-correctness.json)。
- NCU/SASS：终稿的 K1/K2 **272.544/793.440 μs**、H12 K2 **786.528 μs**、候选 K2 **1,583.040 μs** 及资源/吞吐数字与实验报告一致。实际 HMMA/UTCHMMA 证据与“静态计数不是动态指令次数”的限定保留。
- tile：最终数据固定为 [022453478641Z 目录](../results/c1-tile-h96-20260913T022453478641Z/)，源码 SHA-256 为 `878273ce2a4f5b4a4129374a49b45833f592daf2eb9e005557061c9057367e41`。96 CTA 下四组增量和 state update 完整单轮/64 轮时间与 `tile-summary.json` 一致。全 CTA 有限性、issuer-only barrier phase 两项修正已进入终稿，旧方法版本没有被用作最终结论。
- 长序列精度：抽查 [long precision](../results/c1-long_precision-h4-20260913T022617825299Z/precision.json) 四例，官方 output/state 全部 `exact=true`；最大 output 相对 RMSE 为 **0.8166474%**，最差 1024-token 窗口为 **0.8238193%**。终稿四舍五入正确，并说明不同 T 不保证共享前缀。
- 数值机制：C16 的误差 1、C32/64 的部分和溢出、全 FP32 C32 误差 8 / C64 误差 2,147,483,648，以及条件数 128，与数值报告相符。当前工作脚本与实测快照的一次 causal-pair 诊断差别已公开，未将未经重跑的新口径冒充原实测结果。

## 本轮必要修正

1. 将 WRITEUP 的“FP64 / 解析参考约 −7.10e−29”改成明确的解析参考。FP64 三角求解可能将该极小角点算为零，解析值不能冒充 FP64 求解实测值。
2. 在 WRITEUP 与答辩问答定义 **P₈=I−L+L²−…−L⁷**，区分部分和与 **L⁸**；475,020 是 C32 对应部分和的最大幅值。
3. 将普通 dense、非 `.ws`、单 CTA tcgen05 的 N 支持范围写完整为 **8–256 内 8 的倍数**，避免读成没有上界。
4. 将 DEFENSE 的长序列“output 误差”明确为 **相对 RMSE**，不与最大绝对误差混淆；速查表也明确 1.93–2.89× 对应 H96。
5. 两份材料均明确课堂环节的已确认交付方式，并链接本记录；没有使用“已答辩”或虚构现场活动的措辞。

本地 Markdown 链接已逐项检查，目标文件和结果目录均存在。外部 PTX 链接沿用先前已核验的 NVIDIA 官方来源，本轮未重新执行网络可达性检查。以上修正不改变实测数据、性能排序、算法实现或现有证据边界。
