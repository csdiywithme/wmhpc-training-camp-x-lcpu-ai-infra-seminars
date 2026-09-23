# 1.2 修正后验证

2026-09-15；Modal 工作区 simidawhu；NVIDIA B300 SXM6 AC；驱动 580.95.05；nvcc 13.1.80；ARCH=100f。

运行：https://modal.com/apps/simidawhu/main/ap-vLtNP28KrRbSw6pbEqmAXe

## 学员定位与修改

学员观察到 GOT 的对应上下两行完全相同，并指出：“D 的 M 方向重复，说明 A 的 M 方向也就是 row 重复了。”随后本人修改 A fragment 的 a2/a3/a6/a7 装载行坐标，移除此前未声明变量的推导代码。助教仅 review 和复测，未修改题目源码。

结合代码确认根因：这些 fragment 元素本应读取 A 下半部分，却重复读取上半部分；因此错误输出的第 r+8 行复制第 r 行（r=0…7）。原测试数据下部分元素数值偶然相等，所以首次判测是 59/128 处不匹配，而不是 64 处。

## 验证

- 修改前证据：../m1-bug-20260915T121818Z/run.json，FAIL 59/128。
- 修改后课程程序：PASS，退出码 0，128 个输出全部匹配 CPU reference。
- 仅增加 host 打印的诊断副本：PASS，退出码 0；完整输出见 matrix-output.txt。
- 编译均成功；本地快照与远端输入 SHA256 一致。
- 仅验证题目固定输入；未做性能测量。
