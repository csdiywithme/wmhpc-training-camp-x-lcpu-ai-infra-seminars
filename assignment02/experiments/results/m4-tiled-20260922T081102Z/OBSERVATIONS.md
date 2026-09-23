# 4.1 tiled GEMM 验证

2026-09-22，用户重新实现 kernel，助教按请求仅修正 A/B global staging 索引，使其使用全局行跨度 K 与列起点 it*BK。保存的 inputs 是本次真实验证源码，远端 SHA256 与快照一致。

环境：Modal simidawhu，B300 SXM6 AC，driver580.95.05，CUDA13.1.80，ARCH100f。编译 binary/PTX 成功且无诊断输出。测前、测后没有其他 compute process。

| M | N | K | 对拍 |
|---:|---:|---:|---|
|128|64|64|PASS，bad=0|
|128|64|128|PASS，bad=0|
|256|192|256|PASS，bad=0|
|4096|4096|4096|PASS，bad=0|

每次 timeout -k 2s 5s，无GPU自动重试。受限本地网络初次连接失败，没有启动GPU；获准网络运行成功。

4096³普通运行：3.38ms，40.7TFLOPS；同次cuBLAS 1762.4TFLOPS，按打印吞吐计算约2.31%，程序整数打印2%。其余小形状的性能不是吞吐基准。尚未采集NCU或运行Compute Sanitizer，不从低达成率直接断言具体瓶颈。

原始命令、输出、环境与构建记录：run.json；PTX：01_tiled.ptx；运行器：modal_m4_tiled.py。
