# 1.2 首次运行记录

- 日期：2026-09-15；工作区：simidawhu。
- 运行：https://modal.com/apps/simidawhu/main/ap-SQabB9VVRVcfBQt68LRLwv
- GPU：NVIDIA B300 SXM6 AC；驱动 580.95.05；nvcc 13.1.80；ARCH=100f。
- 原题源码未修改；编译成功，执行返回 1，`FAIL: 59 / 128 mismatches`。
- 诊断副本只在 host 增加 got/ref 全矩阵打印，输入、kernel、判测均保留；同样返回 1、59 处不匹配。
- 本地输入快照与远端输入 SHA256 全部一致。
- 原始输出：original-output.txt；完整矩阵：matrix-output.txt；完整构建与运行记录：run.json。

## 学员分析（待填写）

1. D 的错误位置与正确部分有什么关系？
2. 哪个 fragment 的哪部分映射导致该现象？
3. 修改后重新判测的记录。
