# 本轮实验日志

所有时间在具体运行 JSON 中用 UTC 保存；本日志解释设计和决策。没有结果的计划不计为实测。

## 2026-09-16：启动与边界核对

- 用户授权实际探索 C1 / C2，并使用 Modal B200/B300；随后指定可能需要 `simidawhu` 账号。
- 当前 Git HEAD：`3108ba3`。工作区已有个人练习修改和大量旧团队题未跟踪产物；本轮仅在新 `nextgen/` 目录追加，不清理或覆盖旧文件。
- 团队题 README 明确允许完整 AI 实现。个人必做题不在本轮修改范围。
- 已核对现有 C1：上游 `FlashKDA 1ce47ea`、CUTLASS `5c149f5`；现有 split2 负收益不能概括为新指令无效。新实现先验证完整 K2 和 forward。
- 已核对现有 C2：merge-only 的非 PDL 收益与 PDL 退化均保留；新 partial 需要复用冻结验收，不重新校准放宽阈值。
- 本地 `uv` 缓存可运行 Modal 1.5.5；只读取 profile 名，确认包含 `csdiywithme`、`simidawhu`，未输出凭据。
- 最初未显式 profile 的只读 app list 被中断，未启动 GPU。后续所有 Modal 操作显式使用 `MODAL_PROFILE=simidawhu`。

## 首轮待验证假设

### C1

共享状态的双投影可以组成 `H [Kd^T,Qd^T]`；共享 U 的输出修正和状态增量可以组成 `U^T [M^T,Kr]`。可能减少重复供数和等待，但新增 SMEM/TMEM 布局处理和占用可能抵消收益。首先保持各阶段 BF16 舍入边界，复用固定 K1，比对融合与拆分配置。

### C2

QK/PV 的转置映射可以使用逻辑 `(128,16,128)`，但 softmax 的归约方向和 TMEM 结果消费可能增加开销。先建立完整分页计算路径，再测试供数、布局和 split。每 CTA 仅一页的配置没有页间流水机会，不能默认多 stage 更好。

## 连接诊断与静态审查

- `simidawhu` 显式 profile 的初次 app list 失败；两个 API 的 HEAD 探测返回过 HTTP 503，25 秒限时认证连接也超时。这些均为只读调用，没有提交计算。
- 后续 GET API 返回 HTTP 200，Python 直接 TLS 协商 `h2` 成功。Python 系统代理检测发现本机代理；未读取或记录凭据。
- 设置进程级 `MODAL_DISABLE_API_PROXY=1` 后，`Client.from_env` 在约 1 秒内认证成功。没有改系统代理、Modal token 或全局默认 profile。后续命令使用该选项和 `MODAL_PROFILE=simidawhu`。
- 独立静态 review 找到并修正 shared runner 的远端反序列化/本地路径问题，采用 serialized functions；补齐 C2 benchmark 的 candidate_workload 依赖；强制 C2 复用冻结 manifest。C1 原逐位 gate 与新舍入诊断分开，不把新增 allclose 容差称作旧验收 PASS。
- 编译、smoke、verify/bench 的子进程上限分别为 1100、240、950/700 秒；首轮完整 CuTe 编译可能耗时较长，采用 CPU worker，不在 GPU 上等待 nvcc。超时会杀死同进程组子进程。

## 首轮启动失败：Modal 镜像操作顺序

- C2 `20260916T153543276412Z-c2-smoke` 与 C1 `20260916T153831136974Z-c1-smoke` 均已完成共享基础镜像构建（CUDA13.1、Torch2.10.0+cu130、Triton3.6.0、固定 FlashKDA/CUTLASS/FLA），但在进入 local entrypoint 前停止。
- 明确错误：Modal 1.5.5 不允许默认挂载方式的 `add_local_*` 后继续 `.env()` 镜像层操作。退出码均为1，没有提交 CPU candidate function 或 GPU function。
- 修复：将环境设置移到所有本地挂载之前；保留原快照、console与失败状态，新建 run 复验。已完成的基础镜像可复用。
