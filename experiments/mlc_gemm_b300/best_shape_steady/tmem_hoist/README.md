# MMA K 循环外缓存 TMEM 基址实验

已完成：[结果和证据](RESULTS.md)。普通 shared-load 指令减少 85.14%，但配对性能
收益 +0.37% 的区间跨零，没有可靠提速证据。保留原 wide 为默认版本。

用户授权实测前一轮 NCU 离线定位得到的新候选。本实验固定
M×N×K=2048×9472×8192，仅比较当前 wide 与其 TMEM 基址 hoist 变体。
旧 wide 在同卡上作为必要的配对控制，不重跑旧形状 sweep 或旧 NCU 采集。

候选在完成 TMEM 分配和同步后，由参与 MMA 的 elected lane 在 persistent
tile 循环外读取一次 `tmem_addr[0]`，使用显式 let 值构造 MMA TMEM view。
保持 K tile64、有效输出tile512×256、3-stage、128列写回块、148 CTAs、
2-CTA cluster 和其余同步/地址布局不变。写回及释放路径不改。

先在 Modal CPU 容器编译、导出两版 SASS 并检查循环内 LDS；通过后才启动
一次 Modal B300 GPU 调用。GPU 调用自动重试关闭，已有执行禁止重跑。

正式计时协议：同卡、同输入和输出地址、default stream、无显式清缓存；
12 块 ABBA/BAAB，各6块，固定随机顺序；每槽50次预热，300次单调用
CUDA event样本，两方法各7200个正式样本。比较配对块和测量位置，
不以新容器绝对TFLOPS对照旧容器数值判断单变量收益。
事件区间可能包含主机提交间隙，两方法采用相同计时方式。

正确性：FP16输入/输出、FP32累加；3 seeds全输出比较FP32参考再转FP16，
atol=0.01、rtol=0.02；正式计时后两版本再次校验。
正常测速完成后仅为新候选采集一次 NCU（该调用可能由NCU回放多次），
用于核验实际 LDS 变化；NCU时延不用于判定性能提升。

文件：`kernels_hoisted.py` 是候选，`compile_variants.py` 是CPU编译与SASS门禁，
`benchmark.py` 是配对测速和单次profile入口，`modal_runner.py` 管理独立
build/gpu/retrieve阶段及Volume持久化。`runs/` 保存不可变快照、原始样本和报告。
