# C2 候选实现、验收与最终性能

## 固定的最终实现

`candidate.run(case)` 默认使用原始 partial 与新的 feature-tiled merge。原 partial 在文件中仅重命名为 `_page_decode_kernel`；AST 比较含参数、装饰器与函数体完全一致。默认保留原 split 策略、4 warps/3 stages、BF16 概率与 partial、FP8 scale 舍入、两 kernel 结构。

merge 改为 1 warp：S≤8 时每 CTA 写128个输出通道；S16时写64个通道。接口还保留实验用 subpage partial，但最终默认不选择它。策略根据 seed0 探索数据固定，之后用 seeds101/307 独立复测，未根据复测负结果偷偷改变策略。

实现 SHA256：`3e4a2ff87b352c7398ce814cf7ea81cf4e4c1fea33fbb5fe6bec8ee115644773`。

## 探索结果保留

- 首轮小形状15配置：`results/candidate-smoke-20260913T021041Z`，全部正确；该轮TP4 B1 BF16无正收益。
- 全矩阵240配置：`results/candidate-tune-20260913T021243Z`，全部完成；细分token tile有些FP8配置变快，BF16多数不利，没有据此宣称普遍加速。
- 保留原partial、仅改merge的192配置：`results/candidate-merge-20260913T021620Z`，全部完成；据此选择简单的split相关merge策略。
- 早期64-token候选通过过full验收，但该PASS仅属于相应源码快照。最终merge-only再次独立通过同一冻结manifest。

## 完整正确性验收

baseline calibration：168条记录，状态`CALIBRATION_FROZEN`，协议digest `430cf4da832b3fb3d2e3eeb2bff0c31959101a05703cc2ede89a1e10c4c54b02`。
最终heldout：baseline168条、candidate168条，均`PASS`。覆盖三类KV存储、TP1/TP4、B1/4/8/16、物理重排、非恒定scale、stride、tail、dql2/4、空split和padding等冻结域。

候选最大绝对误差 0.009266272；最大逐行NRMSE 0.003973989。这些不是全模型质量指标，也不是所有可能输入的误差上界。

最终paired性能任务还对每行baseline/candidate的实际graph重放后输出做FP64检查：64组均finite且NRMSE<0.02。该补充检查位于计时区间之外，覆盖PDL开/关下实际被测graph；不能替代full数值协议的其他边界。

## 最终测量协议

- 同一B300任务内，相同Q/KV/页表、相同PDL设置、分别预分配workspace，比较完整partial+merge链。
- 每图64次调用，预热后测21个repeat；每repeat随机选择baseline/candidate先后顺序，记录完整顺序和原始样本。
- 所有形状使用独立performance seeds101/307；表格耗时为两个seed各自21样本中位数的算术平均。速度比=同行baseline/candidate，最后列显示两个seed的速度比范围。
- CUDA Graph重复固定地址，无主动清缓存，不等于所有数据都在L2；不包含Python wrapper分配/提交、QKV投影、indexer或服务排队。
- 单卡未锁频；结论是本机型与该协议的证据。21个样本是重复graph平均值，P05/P95不是服务p95，也不被称为统计置信区间。

## PDL=false：对应课程独立harness

|TP|Batch|KV|baseline μs|candidate μs|速度比|两个seed范围|
|---|---|---|---|---|---|---|
|1|1|bf16|5.913|5.363|1.103×|1.099–1.106×|
|1|1|fp8|7.767|7.207|1.078×|1.076–1.080×|
|1|4|bf16|8.277|7.157|1.157×|1.155–1.158×|
|1|4|fp8|10.787|9.763|1.105×|1.104–1.106×|
|1|8|bf16|11.580|9.131|1.268×|1.268–1.268×|
|1|8|fp8|16.638|14.320|1.162×|1.160–1.163×|
|1|16|bf16|15.646|13.546|1.155×|1.155–1.155×|
|1|16|fp8|25.866|23.831|1.085×|1.085–1.086×|
|4|1|bf16|5.975|5.393|1.108×|1.104–1.112×|
|4|1|fp8|7.531|6.943|1.085×|1.078–1.092×|
|4|4|bf16|6.216|5.675|1.095×|1.094–1.097×|
|4|4|fp8|7.991|7.477|1.069×|1.068–1.070×|
|4|8|bf16|6.489|5.899|1.100×|1.099–1.102×|
|4|8|fp8|8.287|7.763|1.068×|1.066–1.069×|
|4|16|bf16|8.300|7.167|1.158×|1.156–1.160×|
|4|16|fp8|10.765|9.769|1.102×|1.102–1.102×|

32个seed/形状结果的速度比范围 **1.066–1.268×**，几何平均 **1.117×**。这对应约6.2%–21.2%延迟降低，不能把速度比减一直接当作延迟降低比例。

## PDL=true：生产相关补充

|TP|Batch|KV|baseline μs|candidate μs|速度比|两个seed范围|
|---|---|---|---|---|---|---|
|1|1|bf16|5.383|6.842|0.787×|0.786–0.788×|
|1|1|fp8|6.952|7.648|0.909×|0.881–0.939×|
|1|4|bf16|7.369|6.156|1.197×|1.193–1.201×|
|1|4|fp8|11.646|10.478|1.111×|1.110–1.113×|
|1|8|bf16|10.875|8.303|1.310×|1.309–1.310×|
|1|8|fp8|19.610|15.234|1.287×|1.284–1.290×|
|1|16|bf16|15.179|12.799|1.186×|1.175–1.197×|
|1|16|fp8|31.685|22.851|1.387×|1.385–1.388×|
|4|1|bf16|5.187|4.721|1.099×|1.097–1.100×|
|4|1|fp8|6.712|6.307|1.064×|1.057–1.072×|
|4|4|bf16|5.408|5.569|0.971×|0.948–0.995×|
|4|4|fp8|7.337|9.997|0.734×|0.724–0.745×|
|4|8|bf16|6.505|6.071|1.072×|1.071–1.072×|
|4|8|fp8|10.800|9.411|1.148×|1.147–1.149×|
|4|16|bf16|7.427|6.208|1.196×|1.196–1.197×|
|4|16|fp8|11.611|10.347|1.122×|1.114–1.130×|

速度比范围 **0.724–1.388×**。TP1 B1及TP4 B4存在退化，不能无条件替换生产PDL路径。该结论来自同卡、相同PDL条件；没有用跨任务时钟差异解释掉负结果。
线程块数、merge资源和PDL重叠会共同影响chain；未采集这些退化形状的完整PDL时间线，故不把某一种调度解释写成定论。默认非PDL实验已显示局部优化值得做，但生产推广需要额外dispatch/PDL策略实验。

## NCU对改动机制的验证

单独的cold NCU使用`cache-control=all`、`clock-control=none`、detailed22passes/kernel。TP1 BF16：

|Batch|merge版本|grid CTA|registers/thread|SMEM actual/ideal wavefront|cold duration μs|
|---|---|---|---|---|---|
|1|原始|64|32|69888/8448|5.216|
|1|候选|128|39|640/640|4.672|
|16|原始|1024|22|577536/86016|6.720|
|16|候选|1024|32|0/0|5.152|

候选B1/B16 merge动态shared分别128/0 bytes，两个case的excess shared wavefront和local spill请求均0。原partial的工作量与资源保持不变。此证据支持减少跨warp布局/归约共享内存开销是改进机制；没有将所有时间差唯一归因于某一个counter。
上述原始/候选NCU来自不同任务；它们支持工作量变化与机制解释，正式速度比以同卡paired表为准。cold NCU时间也不与hot graph表混算。

## 复现与原始证据

```bash
uv --cache-dir /private/tmp/a02-uv-cache run --offline --no-project --with modal==1.5.5 python -m modal run assignment02/team/c2_msa_decode/modal_candidate.py --mode calibrate
uv --cache-dir /private/tmp/a02-uv-cache run --offline --no-project --with modal==1.5.5 python -m modal run assignment02/team/c2_msa_decode/modal_candidate.py --mode verify --manifest assignment02/team/c2_msa_decode/results/candidate-calibrate-20260913T020759Z/calibration.json
uv --cache-dir /private/tmp/a02-uv-cache run --offline --no-project --with modal==1.5.5 python -m modal run assignment02/team/c2_msa_decode/modal_candidate.py --mode paired
uv --cache-dir /private/tmp/a02-uv-cache run --offline --no-project --with modal==1.5.5 python -m modal run assignment02/team/c2_msa_decode/modal_candidate.py --mode profile
python3 assignment02/team/c2_msa_decode/experiments/summarize_candidate.py
```

新校准生成新时间戳路径；运行新verify时替换manifest路径，不覆盖旧manifest。

- [最终实现](../candidate.py)
- [设计与先测后做记录](CANDIDATE_DESIGN.md)
- [验收协议](../validation/ACCEPTANCE.md) / [内部独立审查](../validation/INTERNAL_REVIEW.md)
- [冻结manifest](../results/candidate-calibrate-20260913T020759Z/calibration.json)
- [最终heldout验收](../results/candidate-verify-20260913T022150Z/candidate-heldout.json)
- [最终paired原始样本](../results/candidate-paired-20260913T022747Z/paired.json)
- [候选NCU CSV](../results/candidate-profile-20260913T022426Z/selected.csv)

每个Modal任务保存上传源码快照、命令与GPU环境。失败和探索任务未删除；正式结论只引用这里明确标出的成功最终结果。
