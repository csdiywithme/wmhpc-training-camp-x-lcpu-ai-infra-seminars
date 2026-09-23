# C1/C2 新指令探索：正确性、测量和结论协议

日期：2026-09-16。状态：协议已写入，本文本身不构成任何新候选的 GPU PASS。

用户目标是在 Modal B200/B300 上实际探索新指令性能，并记录思路、实验步骤、过程和结论。本轮覆盖 C1 的完整 KDA forward，以及 C2 的完整 sparse attention partial+merge。tile、单 chunk、单页和 copy 原型是推进阶段；它们不能替代完整实现、相应验收与完整链测量。研究可以得到有证据的负结果，但不能把尚未完成的集成称为性能问题已经解决，也不能宣称有限搜索证明了全局最优。

本文只增加 nextgen 实验约束，不修改既有 C1/C2 源码、gold、验收域和阈值。现有实验结论采用实际保存的源文件及 JSON，不以聊天中的历史概述代替。

## 1. 证据来源及冻结状态

| 项目 | 权威入口 | 本轮使用方式 |
|---|---|---|
| C1 语义、旧结果与局限 | [C1 WRITEUP](../c1_flashkda/WRITEUP.md) | 保留 bounded gate、value-first state、舍入路径和已有负结果 |
| C1 原验证/计时 | [run_experiments.py](../c1_flashkda/run_experiments.py) | 复用官方舍入参考、独立 FP32 naive 语义；改进测量顺序但保留原始数据 |
| C2 原验收政策 | [validation/ACCEPTANCE.md](../c2_msa_decode/validation/ACCEPTANCE.md) | 完整复用 represented-input-v1，不放宽 |
| C2 独立 gold、输入域和不变量 | [suite.py](../c2_msa_decode/validation/suite.py) | 通过 adapter 接入候选，候选不得调用 gold |
| C2 验收 CLI | [run_validation.py](../c2_msa_decode/validation/run_validation.py) | 保留 baseline 前置验证、完整记录计数与 digest 检查 |

C2 本轮应引用原始冻结 manifest：

`../c2_msa_decode/results/candidate-calibrate-20260913T020759Z/calibration.json`

- 状态：`CALIBRATION_FROZEN`；tier：`full`；校准记录：168。
- protocol digest：`430cf4da832b3fb3d2e3eeb2bff0c31959101a05703cc2ede89a1e10c4c54b02`。
- manifest 文件 SHA256：`54783fcd83407338c75d163c99621a33ec71e8b25f75b93ca2f35808c0b7397f`。
- 校准 seeds：11/29；原 heldout seeds：101/307。
- `candidate-verify-20260913T022150Z/calibration.json` 是字节相同的副本；同目录旧 `candidate-heldout.json` 的 168 条 PASS 属于原 merge-only 候选，不属于 nextgen。
- 2026-09-16 静态核验：manifest 的五个源文件 hash 均与当前工作树一致，stored protocol digest 重算一致，168 条 calibration 记录均无失败。这不代表新 GPU 环境的 baseline 或候选已通过。

复用必须在每次远端运行重新检查上传后的源文件和 digest。不得关闭 digest 检查、删去失败 case、修改 manifest 或用新校准结果覆盖它。B200/B300 分别运行 baseline 前置检查；换卡不意味着需要自动重标阈值。若 baseline 失败，保留 `BASELINE_HOLDOUT_FAILED`，先定位环境、输入或实现差异，候选没有取得验收资格。

## 2. 阶段、允许结论与推进条件

| 阶段 | 必须验证的对象 | 可报告的结论 | 不能据此声称 |
|---|---|---|---|
| A：构建与指令 | 实际 cubin/SASS、目标架构、资源用量、入口命名 | 指定二进制使用指定指令，编译/启动成功 | 算法正确或性能更好 |
| B：局部原型 | 全 CTA、全有效输出、结构化输入及随机输入；必要同步和数据转换 | tile/chunk/page 在已测域正确，局部成本多少 | 完整 K2/attention 已完成 |
| C：完整语义 | C1 完整 forward 和末状态；C2 partial+merge 及输出契约 | 对应域与数值口径下的正确性资格 | 服务端到端收益、任意输入正确 |
| D：完整链性能 | 同卡同输入成对计时、原始重复样本、实际计时输出复验 | 该运行域的延迟收益/退化 | 单步 microbench 比例就是整链比例 |
| E：泛化复验 | 冻结实现、未用于调参输入、另一实际 GPU 会话；若声称跨卡则逐卡验证 | 结果在声明的范围可复现 | 有限搜索得到全局最优 |

局部原型可以继续推进，不要求每一步先获得性能收益；但阶段 D 的正式性能资格要求阶段 C 已通过。数值尚未合格的候选可作定位性计时，标签必须为 `DIAGNOSTIC_UNQUALIFIED`，不得进入合格候选排行榜或 dispatch。

布局/指令改动的验证不能只镜像实现：禁止把候选的转置索引、分块循环或 split 合并复制进 gold，然后以一致为证。至少使用一个按数学定义或独立递推组织的参考，配合能揭露索引错误的不变量。

## 3. C1：三种精度问题分别记录

### 3.1 数学语义必须先对齐

- raw gate 使用 `-5 * sigmoid(exp(A_log) * (g_raw + dt_bias))`，beta 使用 sigmoid；q/k 归一化、输出 scale、varlen 边界与原版对齐。
- 公开 state 是 `[N,H,V,K]`；独立 naive 的数学 state 是 `[N,H,K,V]`。D=128 时错误转置不会改变 shape，必须在参考边界显式处理，并用非对称初态和输入检出。
- `FP32 initial/final_state` 是接口类型，不是内部 FP32 持久状态的证明。状态精度消融必须逐一记录跨 chunk 存储、MMA 操作数转换和舍入位置。
- 同一长序列不得切成多个零初态序列当作等价并行；每个输出和末状态均属于语义的一部分。

### 3.2 保留旧 same-rounding gate

当前 C1 `precision(..., candidate)` 对 output 与 final state 使用 `torch.equal`；官方 `torch_ref` 模拟相同近似和舍入。旧 gate 不能替换为 `allclose`，也不能因新硬件改变累计顺序就把失败记为 PASS。

每个 nextgen C1 case 至少记录以下三组比较，output/state 各自分开：

1. 原 baseline 对官方舍入参考：确认当前环境仍重现旧数值路径。
2. 候选对原 baseline/官方舍入参考：记录 finite、bitwise exact、不同元素个数、max abs、RMSE、相对 RMSE；长序列另记位置窗口和首个偏差位置。
3. 候选及 baseline 分别对独立逐 token naive：比较数学误差，不以候选接近 baseline 代替数学检查。现有 naive 内部强制 FP32，不能称 FP64 gold。

官方 reference 不是完全独立的数学 oracle，独立 naive 也不能验证全部实现舍入细节，两者需互补。

### 3.3 tcgen05 无法位等值时的处理

累加组织或浮点结合顺序改变，可能使数学等价的候选不再逐位一致。此时区分两类结果：

- `SAME_ROUNDING_PASS`：旧域全部 output/state 有限并逐位一致，且原 baseline 对官方 reference 的前置检查通过。
- `NEW_ROUNDING_RESEARCH`：明确指出不满足旧位等值 gate；保留失败样例，分别量化实现路径偏差与数学误差。该标签允许继续研究，不能包装成旧验收 PASS。

若希望对新舍入路径设立正式数值合同，必须先单独版本化、列出支持域和参考、冻结误差政策，再在独立输入上验收；不得根据已看到的候选最大误差倒推容差，也不得擦掉旧 gate 失败。当前 C1 旧材料只有位等值 candidate gate，未提供可以直接继承的非逐位候选容差，本文不凭空添加一个“差不多”阈值。

出现第一个差异时，先截取最短触发前缀，分别检查状态投影、残差、U、输出和末状态的数学结果与舍入边界。debug 中间量用于定位；通过同一套 debug 循环不能替代公开入口完整 forward 的验证。性能二进制应去掉调试存储，并重新核验其输出及 SASS/hash。

### 3.4 C1 推进矩阵

- 初级：零输入、非对称小整数/可精确表示输入、单 chunk、两个 chunk、尾 chunk；每个 CTA 使用不同输入，避免只检查 CTA0 或全 CTA 复制同一输出。
- 旧完整精度域：H4；seeds0/1；长度16、17、97、1024、varlen17/33/65；random、weak_decay、strong_decay。复用30组小形状并核验记录数。
- 接口：BF16/FP32 state × initial/final 有/无共8组合，包括非整 chunk varlen。
- 正式性能形状先做 correctness：H12/H96；1×8192、`1300/547/2048/963/271/3063`、8×1024。H64 等新增域独立列出。
- 长链：8192/32768、弱衰减、同一最长序列的多个前缀，记录 output 的1024-token窗口及 final state。若改变持久 state 精度，再加入已有弱衰减/低维 key 的结构性反例，不把不同随机序列两行数据称为误差漂移。

扩展/持久状态实验与兼容路径结果分表。某个结构性输入使 baseline 失败，也必须保存，不能据此直接宣布候选通过原合同。

## 4. C2：原冻结 full gate 完整复用

### 4.1 合同和真实数值门槛

支持域沿用 D128/page128/topk16/GQA16、TP1/TP4、BF16 Q/output、BF16 或 E4M3FN FP8 KV，以及 scalar 或 token/head FP32 scale。gold 先按原路径将 FP8 与 scale 反量化为 BF16 有效 K/V，再升为 FP64 聚齐 attention。它不调用 SDPA、候选、baseline、online softmax 或 split merge。

冻结 manifest 的 `thresholds[storage:pattern]` 是实际准入线。以下是部分关键值；完整以 manifest 为准：

| 族 | max_abs | row_nrmse_max | global_nrmse |
|---|---:|---:|---:|
| bf16:random | 0.0021006776010356605 | 0.02 | 0.02 |
| fp8_scalar:random | 0.0022520727593500762 | 0.02 | 0.02 |
| fp8_token:random | 0.0023262270817197 | 0.02 | 0.02 |
| bf16:sharp | 0.014376452488276403 | 0.02 | 0.02 |
| fp8_scalar:sharp | 0.01434344892743572 | 0.02 | 0.02 |
| fp8_token:sharp | 0.014036416675325125 | 0.02 | 0.02 |
| 其他已冻结族 | 0.002 | 0.02 | 0.02 |

还必须满足 active 输出非有限数为0、逐元素 `abs(error)/(0.02+0.02*abs(gold)) <= 1`，以及原政策全部硬上限。max_abs 的0.02硬 cap、row NRMSE 的0.10 cap、global NRMSE 的0.05 cap 不是可以代替上表的宽松门槛。

FP8 原生计算若额外量化 Q 或 P，仍不能替换 represented-input gold 或改阈值。可以另外展示量化前模型误差、raw-scale FP64 诊断和速度—误差曲线；它们不能取得原 gate 资格。改变精度合同必须单独声明。

### 4.2 候选覆盖与 fallback 透明度

初期仅支持 BF16/dql1/单页，可以接入探索 adapter，但标签只能是该子域。若 adapter 对其他 case 回退 baseline，必须逐 case 记录 `actual_backend`、fast path 是否执行及配置；full gate PASS 只说明整个 dispatch 正确，不能说明未执行的新 kernel 支持全域。

完整实现必须重新覆盖：strided top-k、strided scale、非2幂 scale、打乱物理页、尾页、dql2/4因果边界、空 split、混合/全 padding、Q=0、常量有效 V、sharp、消去和强 causal probe。top-k负值尾槽须按有效长度忽略，scale 用物理 token 下标。

原 metamorphic 三组必须继续执行：有效 top-k 顺序置换、一致物理页重排、合法地址内的未选中/未来数据 NaN poison。变体独立 gold 应先证明数学输入相同；真实共享 prefix 的污染测试需另行构造，不能直接套用 request-private-pages 的 poison 逻辑。

原 full 验收每 adapter 预期168条。不完整输出、异常后提前返回、只有 quick PASS、只有普通随机误差小，都不构成 full PASS。先在同次运行完成 baseline heldout；前置失败时不运行候选验收。

### 4.3 PDL/Graph 与 heldout

原冻结 suite 的 shim 明确关闭 PDL、采用 eager。原 suite PASS 不自动覆盖 Graph 或 PDL。实际计时的 Graph 必须在计时外再次读回同一输出核验；若声称支持 PDL，应将全部支持域在该运行模式另做补充验证，保持 gold 和原冻结阈值。

原 seeds101/307 已公开且反复使用，只能持续承担固定回归检查。若据这些结果调整策略，不能再称其为本轮未见输入。最终候选及 dispatch hash 冻结后，应使用额外预先登记的独立 seeds/形状作复验，沿用已冻结阈值，不以新数据重标阈值。新脚本放 nextgen，不能改旧 suite 以绕过 digest。

## 5. 性能实验：计时范围和公平对照

### 5.1 完整链与诊断分别计时

| 结果名称 | 必须包含 | 单独列出的准备/排除项 |
|---|---|---|
| C1 `forward` | K1、K2、候选必需的转换/打包、实际输出与请求的末状态 | 输入生成、构建/JIT、纯 gold；workspace分配是否包含需说明 |
| C1 `k2_diagnostic` | 真实K1产物和入口状态上的K2完整状态链 | 预计算K1；不能称完整forward |
| C2 `attention_chain` | 物理页解析、必要反量化/转换、全部partial、merge、必需metadata更新 | 生成输入、预热/JIT、gold；热plan和workspace分配明确说明 |
| C2 `page_diagnostic` | QK→mask/softmax→PV、实际必要TMEM/SMEM往返 | 只处理指定页/子域，不能称完整attention |
| `copy`/`tile` | 明确的搬运/矩阵链及同步 | 无完整算子资格 |

依赖输入或每步变化的打包、layout转换、scale/metadata计算必须算入完整链；如果生产确可缓存，明确缓存键、生命周期及不含首次创建的热态口径，同时报告必要的首次准备成本。不能预先为候选计算答案或免除baseline仍承担的工作。

### 5.2 成对测量程序

1. 同一GPU进程中固定输入，baseline与candidate分别完成JIT、分配和预热。记录是否锁频、功耗/温度、device UUID及其他负载；不具备的测量标为未知。
2. 每轮随机或AB/BA平衡顺序，保存顺序和seed。默认至少5个成对重复；每个重复内部使用足够多次重放使区间总时长至少约10 ms，同时设次数上限，避免无限自适应。公开选择规则。
3. CUDA event时间区间必须同stream且在读取前同步。Graph重放与eager分别报告；分母明确是replay数×每graph内算子次数。
4. 每次配置完成后，在计时外检查实际被计时路径的输出和末状态。C1每次重复应使用相同初态语义，不能把上一轮final state偷偷作为下一轮initial state；必要的状态重置成本明确列出。
5. 保存全部原始latency，不只保存最快样本。每形状报告median、离散程度、baseline/candidate配对ratio和绝对节省μs；汇总采用全部预注册形状的几何平均，保留所有退化点。
6. 将正式计时与NCU/时间线分析分开。Profiler中的replay、cache flush和插桩可改变时延；NCU duration不替代正式speedup。

速度比定义为 `baseline_latency/candidate_latency`，延迟降低为 `1-candidate_latency/baseline_latency`。不得把1.20×写成降低20%。多个seed范围不是置信区间；若计算置信区间，记录重采样单位和方法，不能把同一Graph内强相关replay都当独立样本。

### 5.3 选择与复验规则

旧实现和已有优秀候选都作为比较对象：C1原版；C2 vendored baseline及此前merge-only候选。原版和新候选在各自PDL配置下的消融公平性，与实际可用最佳配置之间的比较，都应保留。

只有在对应正确性资格成立、配对收益超过观测噪声、冻结后的复验仍重复出现时，才将某个配置描述为较快。原始数据呈现重叠或反转时标记 `INCONCLUSIVE`，不按最小值选赢家。当前协议不指定任意百分比作为硬发布阈值，也不以几何平均掩盖某些shape的退化。

若采用按形状dispatch，规则在探索集上制定并保存源码hash，随后执行完整回归与独立复验。逐case记录实际backend；不能在看过复验结果后改规则仍沿用“heldout”标签。

B200和B300分别跑同卡对照，并保存各自环境、编译目标和hash。若只跑其中一款，只能报告该卡结果。不同时间/卡的绝对时延差不能直接归因为某条新指令或架构特性。

## 6. GPU生命周期、超时和失败隔离

用户已经授权Modal B200/B300实验。本节用于可追溯的资源控制，不增加逐次用户审批。

- 每次远端function与子进程有显式timeout，启动前记录GPU型号、timeout、候选hash、命令和计划case数；不用自动重试掩盖失败。建议起点为编译600 s、smoke120 s、full correctness600 s、paired benchmark300 s、单profile180 s；若需扩大，先记录具体理由及新上限。这些是每job上限建议，不是已获准或已经消耗的总GPU预算。
- 先编译/启动smoke，再大矩阵。trace/sanitizer仅覆盖有明确问题的代表点，然后按修改范围复验；不为产生更多图表无限增加GPU工作。
- shell/观察超时不代表远端job结束。必须检查已有Modal app/function/job handle，确认live/terminal再决定是否继续；禁止因一次轮询失败重复启动同一实验。
- 对异步非法访问、device assert、死锁或显式timeout，立即保存首个错误、stderr、退出码和已完成case；停止该CUDA进程，后续case放新进程。CUDA context损坏后的连锁异常不算多个独立失败。
- timeout或进程异常同样是实验记录，状态分别为 `BUILD_FAILED`、`RUNTIME_FAILED`、`TIMED_OUT`、`INCOMPLETE`；不能因没有正确性数字写PASS。保留失败源码与日志；修复使用新run ID，不覆盖。
- 每个结果目录包含UTC启动/结束、Modal app/function ID、GPU实际运行秒数（可得时）、wall time、输出下载路径、终止状态。wall time与编译/排队时间不是GPU计费秒数；无法读取费用时写未知，不猜金额。
- 实验阶段结束必须核实没有遗留运行/重试任务。主动取消应记录取消对象和原因，保留已产生输出。

## 7. 每次实验的最小记录

每个不可变run目录至少保存以下内容，日志先落盘再生成摘要：

| 类别 | 必须记录 |
|---|---|
| 假设 | 预期消除的成本、可能引入的成本、能证伪假设的观察 |
| 变量 | baseline与candidate仅有哪些变化、控制项、探索参数范围 |
| 可复现输入 | 形状、dtype/stride、gate/scale、seed、模式、case ID；必要时保存最小失败张量 |
| 源码 | candidate/adapter/runner/gold/source hashes、git状态、上传快照、实际编译命令和架构 |
| 环境 | GPU名称/CC/SM数/UUID、驱动、CUDA toolkit、Torch/Triton/CUTLASS版本、编译器flags |
| 数值 | 全部case记录、对应reference合同、实际backend、错误分类、预期/完成数、原冻结manifest hash |
| 性能 | 计时范围、PDL/Graph/cache/alloc策略、预热/重放数、测量顺序、全部样本、正式汇总 |
| 机制 | 实际SASS与资源、代表性profiler数据；不将静态指令计数解释为动态耗时比例 |
| 过程 | 原始stdout/stderr、退出码、timeout、Modal句柄、时长、修复与下一步理由 |
| 结论 | 假设支持/反驳/证据不足、允许外推的域、尚未完成项、继续或停止该路线的理由 |

## 8. 本轮目标完成审查

结束探索时，C1与C2必须各自回答：实际实现了哪些新指令数据路径；有没有接到完整算子；完整语义/精度验证结果；同卡完整链收益或退化；最关键的机制证据；另一GPU是否实测；未解决的数值、资源或集成障碍。每项链接到不可变run证据。

以下均不是完成证明：仅有设计文档；只编译成功；仅单tile/page变快；使用fallback完成全部case却未说明新kernel覆盖率；旧结果目录已有PASS；只跑baseline；略去数值失败后比较性能；只有云任务启动记录而无终止与输出。

若完整原型在准确实现后稳定变慢，应记录具体负结果和至少能区分主要解释的对照，而非强行选择最有利局部数字。若仍缺必要实现、验收或完整测量，则将它明确列为未完成，并继续目标，不把子阶段成功替代用户授权的整体探索。
