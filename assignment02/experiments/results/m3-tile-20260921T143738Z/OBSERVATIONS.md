# 3.2 教学实现验证

2026-09-21：用户明确要求补全代码后逐段学习。实现位于 cuda/m3_tcgen05/02_single_tile.cu。

- Modal B300 SXM6 AC，CUDA 13.1.80，driver 580.95.05，ARCH=100f。
- 课程原始 judge_tile.sh：seed 1、7、42、1234、99999 全部严格对拍 PASS（每次 8192 个输出）。
- compute-sanitizer memcheck 和 synccheck：seed 42 均 0 errors。
- run.json 保存命令、输出、编译结果与输入哈希；inputs 保存实际源码；02_single_tile.ptx 保存 PTX。
- 首轮额外 seed 0、1、42、123、2026 的 PASS 见 ../m3-tile-20260921T143616Z/run.json。
- 读回采用 .32x32b.x1，每次一列，便于学习；没有测量或声称性能优化。
- 题目要求的去掉 proxy fence 的观察实验尚未执行，留待逐段学习后进行；不能从正确版本 PASS 推断省略 fence 合法。
