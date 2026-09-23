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

## 第二轮启动失败：序列化 Python 版本

- `20260916T154137351229Z-c2-smoke` / `20260916T154137425311Z-c1-smoke` 在函数注册阶段发现本地 Python3.13 与镜像 Python3.11 不兼容；均退出1，未提交候选函数或GPU。
- 本机已有 Python3.11.15，后续明确以 `uv run --python 3.11 --with modal==1.5.5` 启动。runner 加入口版本检查，避免再次完成远端镜像后才发现版本错误。所有失败日志保留。

## 第三轮 CPU 容器初始化失败与导入方式修复

- `20260916T154338270903Z-c2-smoke` / `20260916T154338353251Z-c1-smoke` 已获得 CPU function call ID，但 serialized function 的 helper 仍引用 `modal_runner` 模块，远端找不到，候选编译未开始。
- 平台对容器启动失败仍会重启，即便函数 `retries=0`；发现重复异常后主动 stop 两个 app，保存调用句柄及取消状态。没有 GPU function 提交。
- 改为标准文件导入：全部本机路径/镜像构造放在 `modal.is_local()` 分支，远端只定义函数，使用上传的冻结协议目录。去除 serialized=True，避免 helper 的按模块引用问题。
- 本地模拟远端（无 NEXTGEN_RUN_DIR/profile）的完整模块导入已成功；独立审查对 Modal1.5.5 自动脚本挂载、image=None 注册与函数导入源码逐项核验。下一轮先确认 C2 CPU 进入 nvcc 再推进 C1。

## 首次真实候选编译：C2 常量头文件遗漏

- `20260916T154654373706Z-c2-smoke` 已成功运行 CPU function 并下载 `build.json` 与编译目录归档，worker 源码路径/hash一并保存。
- nvcc 用时约21.84秒，失败点是 `partial.cu` 三处 `CUDART_INF_F` 未定义。没有提交GPU。将补齐 CUDA math constants 的显式依赖；新源使用新run，原失败保留。
- 此时才启动 C1 `20260916T154757438758Z-c1-smoke` 的 CPU 编译，避免在执行链路不通时持续叠加任务。

## 首轮 GPU smoke 完成

- C1 `20260916T154757438758Z-c1-smoke`：CPU编译73.45秒，K2 174寄存器、0spill。B300上3例完整forward输出/末state与旧版位等值，官方舍入前置亦通过；状态是 `SMOKE_SCOPE_SAME_ROUNDING`，不冒充完整38例资格。独立FP32 naive误差与baseline相同；workspace oracle有一例输出2元素不一致，最大1.52587890625e-5；该辅助参考使用不同tanh近似及GEMM累计组织，原因尚未逐项隔离，保留差异。
- C2 `20260916T154921959378Z-c2-smoke`：补头文件后CPU编译42.33秒，BF16/FP8分支均70寄存器、0spill。15例（5输入×splits1/4/16）完整partial+merge在eager和实际Graph后通过硬cap smoke；最大实测绝对误差约0.0009683。该阶段未执行full frozen gate，不能宣称full PASS。
- 两个app均正常完成，原始stdout/stderr、二进制资源、SASS、输入域与数值JSON已下载。下一步校验compile-source及bundle hash后复用二进制，分别执行C1 30旧域+8接口、C2冻结full manifest验证。

## 首轮完整正确性通过

- C1 `20260916T155241003026Z-c1-verify`：38/38（旧30+8接口）全部输出/state位等值，状态 `SAME_ROUNDING_PASS`，子进程58.84秒。未发现新舍入偏差；这不表示相对独立数学参考误差为零。
- C2 `20260916T155241079346Z-c2-verify`：同次baseline168/168、新candidate168/168均PASS；复用原digest `430cf4da...`，无重校准；子进程72.67秒。adapter审计记录新partial实际执行，无fallback。
- 两个app正常完成。首轮正式bench启动：C1六形状，5轮平衡AB/BA完整forward及诊断K2；C2 2TP×4batch×2storage×2seed=32组，7轮随机顺序比较baseline、旧merge-only、新partial+同merge。每轮按约10ms选择迭代/图重放数，保留全部样本。

## C1 首轮完整性能：正确但显著变慢

- `20260916T155457579434Z-c1-bench` 六组完成且计时域符合旧舍入资格。速度比（旧版延迟/候选延迟）依次为0.06328、0.06866、0.08644、0.07655、0.07068、0.06717，几何平均0.07175。
- H96单序列8192：旧版1076.99µs，新版14069.77µs；禁用K1的诊断调用13782.85µs，主要问题在新K2路径。诊断调用仍包含binding及beta准备，不称纯硬件kernel duration。
- 174寄存器但0spill，不能把退化归因于溢出。首版约160KiB共享存储、FP32 scratch及value-first重排是候选解释；bank conflicts与访存合并情况尚未由profiler确认。
- 下一轮重点：在寄存器直接消费TMEM结果，去掉FP32共享中间区，保留BF16舍入边界；同时用地址映射/单点profile检查布局成本。先保存当前完整负结果，不以tile吞吐替代它。
