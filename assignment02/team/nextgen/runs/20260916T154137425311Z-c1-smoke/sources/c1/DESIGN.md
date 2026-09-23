# C1 完整 tcgen05 K2：首版设计与实验入口

日期：2026-09-16。首版代码已完成本地 Python 语法与生成 launcher 检查；本文不构成 CUDA 编译、GPU 正确性或加速证明。实际编译/运行状态以 `assignment02/team/nextgen/results/` 下不可变 run 的原始日志为准。

遵守 [本轮冻结协议](../../nextgen/ACCEPTANCE.md)。上游 `FlashKDA/`、旧报告与旧数值门槛保持不变。

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

当前尚未单独加入固定同前缀的 8192/32768 长链、全部结构性反例、每阶段 debug trace、独立 heldout 和第二 GPU 复验。这些是后续必要的外推边界，不由38例替代。改变内部持久状态精度是另一条研究路线，当前首版没有做该改变。

## 测量和后续实验顺序

当前 runner 使用 CUDA events 测 eager 完整 forward，workspace 已分配，包含 K1、beta 转置和完整 K2；输入生成、编译、纯 reference 不计时。baseline/candidate 以 AB/BA 顺序交替，保存每次原始样本。每次调用重新读取同一个初态，不把上次末态偷渡成下一次初态。K2-only 分列，计时后读回实际输出/末状态。

初步顺序：构建与 SASS → 小形状 smoke → 38例精度 → 首个差异定位或完整链计时 → 四种融合消融。只有原型的语义和瓶颈已明确后，才尝试消除大 shared FP32 scratch、直接 TMEM 消费、TMA 双缓冲或 value-column split。异步流水优化需要重新验证 barrier phase、buffer 生命周期与全部输出，而不能仅验证 CTA0。

现有正式形状是 H12/H96 × 单条8192、六条不等长总8192、8×1024；quick 形状只是启动成本和早期退化定位。CUDA Graph、至少约10ms的自适应测量窗口、设备功耗/频率记录、最终冻结 heldout 等更严格测量由根 runner/后续补充完成；不能把当前 eager 样本直接称为部署最优延迟。
