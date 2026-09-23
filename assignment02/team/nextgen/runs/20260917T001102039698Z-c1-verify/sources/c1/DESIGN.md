# C1 完整 tcgen05 K2：首版设计与实验入口

日期：2026-09-16/17。首版已经完成真实 SM103a 编译、B300 的3例 smoke、38例精度域、4例同前缀长链与六形状正式计时；首版性能明显退化。下文明确各项已测范围，v2尚待GPU证据。实际编译/运行状态以 `assignment02/team/nextgen/runs/` 下不可变 run 的原始日志为准。

遵守 [本轮冻结协议](../../nextgen/ACCEPTANCE.md)。上游 `FlashKDA/`、旧报告与旧数值门槛保持不变。

## 首轮已观察证据

[首轮 build](../../nextgen/runs/20260916T154757438758Z-c1-smoke/build.json) 使用固定 FlashKDA `1ce47ea3`、CUTLASS `5c149f52`，真实 CPU worker 编译约73.445秒。K2 ptxas 报告174个寄存器、0 stack、0 spill、1 barrier；[cuobjdump 资源记录](../../nextgen/runs/20260916T154757438758Z-c1-smoke/artifacts/binary-0-resources.json) 与此一致。资源记录中的 static shared 不含 launch 时申请的动态 shared，不能把 `SHARED:1024` 当作整个 kernel 只用了1KiB。[实际 SASS](../../nextgen/runs/20260916T154757438758Z-c1-smoke/artifacts/binary-0.sass) 的 K2 中出现 `UTCHMMA`。

[B300 smoke](../../nextgen/runs/20260916T154757438758Z-c1-smoke/artifacts/smoke-fused.json) 状态是 `SMOKE_SCOPE_SAME_ROUNDING`：H2、T17；H2、T33无初态；varlen1/17/33且FP32初末态，三例输出与末状态全部和原版逐位相同，baseline 官方参考前置也全部通过。候选与原版分别对独立 FP32 naive 的误差逐项一致。

辅助 workspace Torch oracle 在第二例输出有2个元素不同，max abs为1.5258789e−5，相对RMSE为4.54e−5；末状态仍一致。该辅助参考使用不同的 `torch.tanh` 与FP32 GEMM组织，目前没有将差异单独归因于其中某一项，不能把它冒充旧官方舍入参考。候选对旧官方/baseline位等值不受这条辅助诊断影响。

以上仍然不是长链/全域资格或性能收益证明。首轮 kernel source SHA256为 `542719ee5e7e6e0efcdeed3b158dab1423c06cf1f75fd7256ecba2499a78b062`，后续消融必须保存各自实际源码与二进制。

后续 [38例完整精度记录](../../nextgen/runs/20260916T155241003026Z-c1-verify/artifacts/verify-fused.json) 实际完成旧30例与8接口，状态 `SAME_ROUNDING_PASS`。独立 [长链记录](../../nextgen/runs/20260916T155933523315Z-c1-verify/artifacts/long-verify-fused.json) 的H4、seed907、random/weak_decay、8192/32768共4例也和baseline输出/末状态逐位相同，状态单独标为 `LONG_SCOPE_SAME_ROUNDING`；来源是一条32768输入的共享前缀。对FP32 naive的output相对RMSE：random约0.5251%/0.5252%，weak_decay约0.7757%/0.7827%；这些是原版与候选共同的整实现误差，不是状态精度消融结果。

[六形状正式计时](../../nextgen/runs/20260916T155457579434Z-c1-bench/artifacts/bench-fused.json) 全部为负收益。五个配对重复的median，单位μs：

| H | 序列 | baseline forward | fused forward | baseline/fused |
|---:|---|---:|---:|---:|
| 12 | 1×8192 | 835.14 | 13,198.09 | 0.0633× |
| 12 | 六条不等长总8192 | 356.76 | 5,196.25 | 0.0687× |
| 12 | 8×1024 | 154.03 | 1,781.99 | 0.0864× |
| 96 | 1×8192 | 1,076.99 | 14,069.77 | 0.0765× |
| 96 | 六条不等长总8192 | 886.53 | 12,543.53 | 0.0707× |
| 96 | 8×1024 | 699.34 | 10,411.35 | 0.0672× |

六形状几何平均速度比约0.07175×。H96单长序列的candidate K2-only约13,782.85μs，说明主要增量确实位于新K2，而不能用K1解释。但这还不能把全部损失归因于某一条内存指令或tcgen05本身。

已发现的静态证据是TMEM读回按lane对应输出行，而首版FP32 scratch是row-major：N32/16/144阶段对shared的行跨度分别128/64/576字节，实际SASS生成 `STS.128`。必须按128B子事务分析，而非直接把标量bank算式称为16/32-way实测；对应静态推导为N32约8-way、N16/144约4-way，需要NCU反证/确认。首版V读取与输出写回的lane沿token变化，warp内16对BF16跨16个sector，存在明显不合并访问。初版174寄存器且0spill，不能把退化归因于spill。

## 研究问题与可证伪假设

旧 tile 微基准的 `128×128×16` 增量优势不能推出完整 K2 更快。本实验必须包括真实 K1 工作区、跨 chunk 的 BF16 状态、beta/残差/U 舍入、输出与末状态。

假设：将共享操作数的两条独立支路合并，可以减少瘦矩阵 tcgen05 操作的提交/等待和操作数准备成本；其收益可能被 TMEM 结果消费、shared footprint、寄存器压力和无流水的初版搬运抵消。融合后完整链仍慢是有效负结果，但必须先排除不正确实现。

首版以一个 CTA 处理一条序列的一个 head，C16、D128。它是保守的完整语义原型：暂未实现 TMA 输入预取、warp specialization、跨 chunk 双缓冲、value-column split 或跨 CTA 协作；这些均不能写作已经完成的优化。

## 计算图与数据流

状态采用公开接口的 value-first 方向 `H=Sᵀ`，形状 `[V,K]=[128,128]`，持久保存在 shared memory，初始 FP32 接口先转 BF16。

| 阶段 | 数学计算 | tcgen05 M×N×K | 必须的数值边界 |
|---|---|---|---|
| 状态投影 | `H [K_dᵀ, Q_dᵀ]` | `128×32×128` | 两个投影分别转 BF16 |
| 残差/校正 | `Eᵀ=((V−K_dS)⊙β)ᵀ`；`Uᵀ=EᵀRᵀ` | `128×16×16` | BF16 减法、BF16 乘法；U 转 BF16 |
| 输出/状态增量 | `Uᵀ [Mᵀ,K_r]` | `128×144×16` | 输出修正先转 BF16，与 BF16 基础输出相加；状态增量保留 FP32 |
| 状态提交 | `H'=H⊙exp(G_C)+ΔH` | 标量 epilogue | FP32 FMA 后转 BF16；gate 沿 key 列广播 |

所有右侧因子在 shared 中按普通 `B[N,K]` 顺序准备，再复制到 CUTLASS UMMA 所需 swizzle。tcgen05 以 shared/shared 操作数执行；FP32 累加器位于 TMEM，随后由四个 warp 读到寄存器并写入普通 shared scratch，便于明确验证逐元素处理。首版不是“TMEM 内全融合”：结果消费与转换成本确实存在，并计入 K2。

分配一次 256 列 TMEM，供各阶段串行复用。每阶段的 issuer warp 推进完成 barrier phase，其他 warp 通过 CTA 同步进入消费阶段，消费完再次同步才复用缓冲。没有每条 K16 都单独等待；等待只出现在完整 GEMM 的结果依赖处。初版 `acc` scratch 较大，TMEM 读出还可能产生寄存器/SMEM 压力；实际资源由 ptxas/cuobjdump/NCU 记录。

当前 chunk 的 `E→U→H'` 是真实依赖链。下一 chunk 在完整 BF16 状态提交后开始，不能通过异步指令跳过递推依赖。输出只写本序列有效 token，varlen 使用原 K1 tile prefix，不将长序列切成零初态子序列。

## 源码与变体

- `k2_tcgen05.cuh`：完整多 chunk recurrence，包括 BF16/FP32 状态接口、无初态/无末态、固定长度/varlen、tail。
- `build.py`：复制原绑定和 launcher 到指定构建目录；只替换 K2 launch，并加可关闭 K1 的诊断开关。不会写上游树。明确 `--arch`，可在无 GPU CPU 构建 worker 编译。
- `run.py`：同输入原版对照、官方舍入参考、独立逐 token FP32 naive、辅助 K1-workspace Torch K2 参考；完整 forward 与单独 K2 诊断计时。

| `--variant` | 首投影融合 | 输出/状态融合 | 用途 |
|---|---:|---:|---|
| `fused` | 是 | 是 | 首选原型 |
| `split-qk` | 否 | 是 | 区分首阶段融合成本 |
| `split-final` | 是 | 否 | 区分末阶段融合成本 |
| `split-both` | 否 | 否 | 同新指令、同完整数据流的未融合对照 |
| `baseline` | 原实现 | 原实现 | 带 K1 开关的原版，用于 K2-only 公平诊断 |

v2在独立 `k2_tcgen05_direct.cuh`，保留首版文件与不可变run。增加 `--variant direct-rowmajor` 与 `--variant direct`：两者都将TMEM值读入寄存器后按CuTe identity坐标执行epilogue，删除73,728B FP32 shared scratch，并使V读取/输出store随lane的value维连续；前者保留原BF16 state/U/base backing，后者再将这些缓冲的value维改为连续。这样可用首版→direct-rowmajor区分直接消费的效果，用direct-rowmajor→direct区分持久/中间BF16布局的效果；它们都必须重新编译、38例验证和同卡计时，不能继承首版PASS。

编译与宽读取的独立消融入口如下，详情与实际结果见 [PROFILE_ANALYSIS.md](PROFILE_ANALYSIS.md)：

| 变体 | header | `-DNDEBUG` | `C1_TRANSPOSE_SHARED` | TMEM load |
|---|---|---:|---:|---:|
| `direct` | `k2_tcgen05_direct.cuh` | 否 | 1 | 1x |
| `direct-release` | 同上 | 是 | 1 | 1x |
| `direct-wide` | `k2_tcgen05_direct_wide.cuh` | 是 | 1 | 16x |
| `direct-wide-rowmajor` | 同上，逐字相同 | 是 | 0 | 16x |

`direct-wide-rowmajor` 与 `direct-wide` 只有shared物理布局宏不同，用于在相同release/宽load条件下复验布局取舍。两个wide变体都执行固定CUTLASS原语审计，必须保存各自manifest、重新取得本变体38例资格；`run.py`继续严格匹配variant与全部nvcc flags，不允许复用另一个布局的PASS。

`set_k1_enabled` 是进程级诊断开关，不是线程安全生产 API。首个完整 forward 填好 workspace 后才允许关闭 K1。K2-only 仍包含 beta 转置和 host launch setup，但不含预计算的 K1；不能称为完整 forward。

## 构建/运行命令

远端依赖：已固定的 `/opt/FlashKDA` 与其 CUTLASS 子模块、CUDA 13.1、Torch 2.10、ninja/C++ 编译器。构建目录中的 `.so` 与 `manifest.json` 必须一起保存/传递。

```bash
python /opt/nextgen/c1/build.py --flash-root /opt/FlashKDA --arch 100a \
  --output /tmp/c1-fused --variant fused
python /opt/nextgen/c1/run.py --build-dir /tmp/c1-fused --mode smoke \
  --output /tmp/c1-smoke
python /opt/nextgen/c1/run.py --build-dir /tmp/c1-fused --mode verify \
  --output /tmp/c1-verify
```

B300 用 `--arch 103a`；B200 用 `100a`，逐卡保存实际 GPU 信息，不把两卡差异归因于某条指令。

正式计时要求同源码、同变体完整精度资格文件；没有资格时只允许显式诊断模式：

```bash
python /opt/nextgen/c1/run.py --build-dir /tmp/c1-fused --mode bench \
  --suite formal --qualification-json /tmp/c1-verify/verify-fused.json \
  --output /tmp/c1-bench

python /opt/nextgen/c1/run.py --build-dir /tmp/c1-fused --mode bench \
  --suite quick --allow-unqualified-timing --output /tmp/c1-diagnostic
```

后者必须标记 `DIAGNOSTIC_UNQUALIFIED`，比值字段为 `diagnostic_latency_ratio`，不进入合格候选排行。需要原 K2-only 对照时，另编译 `--variant baseline` 并传 `--baseline-build-dir`。

CPU 本地不编译 CUDA 的生成审计命令：

```bash
python assignment02/team/c1_flashkda/nextgen/build.py \
  --flash-root assignment02/team/c1_flashkda/FlashKDA --arch 100a \
  --output /private/tmp/c1-nextgen-static --variant fused --generate-only
```

## 正确性口径与当前覆盖边界

`smoke` 有 3 例，含多个 chunk、tail、varlen 与 FP32 状态接口；每例先要求原 baseline 对官方 `torch_ref` 位等值，否则候选无资格。它只产生 `SMOKE_SCOPE_SAME_ROUNDING` 或研究/失败状态，不产生完整 PASS。

`verify` 直接复用旧 `run_experiments.inputs` 的 30 组输入：H4；seed0/1；长度16、17、97、1024、varlen17/33/65；random、weak_decay、strong_decay。额外8组覆盖 BF16/FP32 接口 × 初态/末态有无。每例分别记录：baseline→官方参考、candidate→baseline/官方参考、两者→独立 FP32 naive。naive 不是 FP64 gold，workspace Torch 参考也不独立验证 K1。

旧门槛仍是 finite + `torch.equal`；不存在新设的 allclose/RMSE PASS 线。不位等值会保存第一组 `first-difference.pt`，并完整记录误差与 `NEW_ROUNDING_RESEARCH`。只要出现非有限数或 baseline 官方参考前置失败就停止当前进程。

当前尚未加入全部结构性反例、每阶段 debug trace、最终冻结后的独立 heldout 和第二 GPU 复验。这些是后续必要的外推边界，不由38例或上述4例长链替代。改变内部持久状态精度是另一条研究路线，当前首版没有做该改变。

另有独立长链入口 `--mode verify --suite long`，不改变旧38例。默认 seed907、H4、长度8192、random/weak_decay；用旧 `inputs()` 一次生成32768源序列，从同一组 q/k/v/g/beta 与初态截取前缀，因此分开跑8192与32768仍能比较共享前缀。每例记录 candidate↔baseline 输出/末状态exact、两者↔固定 FP32 naive、1024-token输出窗口及首差异位置；不重新运行长链官方舍入参考，也不会产生旧 `SAME_ROUNDING_PASS`，只产生独立的 `LONG_SCOPE_SAME_ROUNDING` 或研究/失败状态。

```bash
python /opt/nextgen/c1/run.py --build-dir /tmp/c1-fused --mode verify \
  --suite long --lengths 8192 --gates random,weak_decay --seed 907 \
  --source-length 32768 --output /tmp/c1-long-8192

python /opt/nextgen/c1/run.py --build-dir /tmp/c1-fused --mode verify \
  --suite long --lengths 32768 --gates weak_decay --seed 907 \
  --source-length 32768 --output /tmp/c1-long-32768-weak
```

输出文件是 `long-verify-fused.json`。`--lengths 8192,32768` 可在同一次运行比较两前缀；为限制GPU任务时长，可每次仅运行一个长度/一个gate。该入口是已实现的检查工具，是否已运行仍必须查看实际run证据。

单点NVTX入口供Nsight Compute使用（只捕获一次完整forward，预热和正确性比较在范围外）：

```bash
ncu --target-processes all --nvtx --nvtx-include c1_nextgen_profile_tcgen05/ \
  --set detailed --export /tmp/c1-profile \
  python /opt/nextgen/c1/run.py --build-dir /tmp/c1-fused --mode profile \
  --profile-target tcgen05 --profile-heads 96 --profile-length 8192 \
  --output /tmp/c1-profile-artifacts
```

原版对照改 `--profile-target baseline` 与 `--nvtx-include c1_nextgen_profile_baseline/`。结果只标 `PROFILE_DIAGNOSTIC_COMPLETED`，NCU时间不替代正式成对计时。NVTX范围包括K1和K2；分析时按实际kernel名称分开。

## 测量和后续实验顺序

当前 runner 使用 CUDA events 测 eager 完整 forward，workspace 已分配，包含 K1、beta 转置和完整 K2；输入生成、编译、纯 reference 不计时。baseline/candidate 以 AB/BA 顺序交替，保存每次原始样本。5次调用的 pilot 估计较快实现的单次时间，然后以 `ceil(10000/fastest_us)` 选择调用数，默认下限30、上限1000，使每个独立样本区间目标约10ms；记录 pilot、上下限、chosen 与实际区间长度，命中上限不伪称达到10ms。每次调用重新读取同一个初态，不把上次末态偷渡成下一次初态。K2-only 分列，完整 forward 与 K2-only 计时后分别读回实际输出/末状态。

初步顺序：构建与 SASS → 小形状 smoke → 38例精度 → 首个差异定位或完整链计时 → 四种融合消融。只有原型的语义和瓶颈已明确后，才尝试消除大 shared FP32 scratch、直接 TMEM 消费、TMA 双缓冲或 value-column split。异步流水优化需要重新验证 barrier phase、buffer 生命周期与全部输出，而不能仅验证 CTA0。

现有正式形状是 H12/H96 × 单条8192、六条不等长总8192、8×1024；quick 形状只是启动成本和早期退化定位。CUDA Graph、设备功耗/频率记录、最终冻结 heldout 等更严格测量由根 runner/后续补充完成；不能把当前 eager 样本直接称为部署最优延迟。
