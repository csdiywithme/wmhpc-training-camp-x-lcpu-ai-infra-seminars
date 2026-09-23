# 1.3 FP8 MMA 验证

日期：2026-09-16。Modal 工作区 simidawhu；NVIDIA B300 SXM6 AC；驱动 580.95.05；nvcc 13.1.80；ARCH=100f。

运行：https://modal.com/apps/simidawhu/main/ap-9DFAEVfLj2AMZtgCSPskf5

学员编写了 A/B 手工装载、MMA 和写回主体。host main 由助教按请求提供。本轮经学员明确要求，助教修正 D 坐标映射，改用 memcpy 打包四个 FP8 的原始编码，将映射函数移到 host/device 共用头文件，并移除无定义的 extern 声明及未完成的类型别名。

课程 judge_mma_fp8.sh 原样执行：seed 1、7、42、1234、99999 全部 PASS，各 128 项严格匹配，最终 JUDGE: PASS。编译无错误。输入快照与远端 SHA256 一致。

提取头文件后 1.1 的本机 clang++ host 回归为 PASS。没有测量性能，没有在 5090 上运行。
