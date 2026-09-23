# 3.3 原始版本：短超时复现

2026-09-22，B300 SXM6 AC，CUDA 13.1.80，driver 580.95.05，ARCH=100f。
题目 CUDA 源码未改；保存于 inputs，哈希已与远端核对。

沿用 judge_mbar.sh 的 rounds={1,2,4} 与 seed={42,7} 组合，但根据用户要求，将每用例 30 秒缩短到 timeout -k 2s 5s。GPU 函数上限 90 秒，无重试。

| rounds | seed 42 | seed 7 |
|---|---|---|
| 1 | PASS, exit 0 | PASS, exit 0 |
| 2 | TIMEOUT, exit 124 | TIMEOUT, exit 124 |
| 4 | TIMEOUT, exit 124 | TIMEOUT, exit 124 |

超时用例 stdout/stderr 均为空。这里只能确认程序未在 5 秒内完成；没有进行 PC 定位，不能把超时直接等同于卡在某条具体指令。
run.json 含完整命令、输出、编译记录；03_bug_mbarrier.ptx 为本次生成 PTX。
后续：学员分析 phase/arrival 状态转换后修复，再对拍和比较。未修改题目源文件或原 judge。
