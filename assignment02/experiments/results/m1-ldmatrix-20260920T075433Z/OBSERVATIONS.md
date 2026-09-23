# 1.4：手工装载与 ldmatrix

2026-09-20，用户明确请求助教完成并讲解。助教实现两个 loader，使用此前学员通过 1.1 的 A/B 映射；保留给定 CPU reference 与判测。

## 实测证据

- Modal 工作区 simidawhu，NVIDIA B300 SXM6 AC，driver 580.95.05。
- CUDA 13.1.80，`make -B ARCH=100f bin/m1_sm80/04_ldmatrix ptx/m1_sm80/04_ldmatrix`。
- seeds 1、7、42，每个 seed 两条路径的 128 个输出均与 CPU 严格相等，所有 mismatches=0。
- [运行记录](run.json)、[判测输出](judge-output.txt)、[PTX](04_ldmatrix.ptx)、[逐条指令分类](instruction-counts.json)。inputs/ 保存真实源码快照，运行器核对其 SHA256 与远端源码一致。
- Modal app: https://modal.com/apps/simidawhu/main/ap-VzZgDLRCTXXWY6dszAPCYo （已结束）。
- 最初离线启动因临时依赖缓存失效失败，恢复联网下载 CLI 后成功；该失败发生在 GPU 启动之前。

## 地址与 fragment

`m8n8.b16` 的一块数据是 8 行 × 16 字节；b16 是原始 bit 搬运，每个 b16 包含两个 FP8，无浮点格式转换。非 trans 模式下，接收 lane 的局部行是 lane/4，行内字节起点是 4*(lane%4)。每个 tile 向每个 lane 写一个 b32。

A 为 sA[16][32]，四块按左上、左下、右上、右下顺序进入 a[0..3]：

| 提供地址的 lane | 行 | 起始字节列 | 目标寄存器（所有 lane） |
|---|---|---|---|
| 0..7 | 0..7 | 0 | a[0] |
| 8..15 | 8..15 | 0 | a[1] |
| 16..23 | 0..7 | 16 | a[2] |
| 24..31 | 8..15 | 16 | a[3] |

因此地址为 `sA + (lane & 15)*32 + (lane >> 4)*16`，使用 `.x4`。
例如接收 lane 0 的 a[0..3] 依次含 A[0][0..3]、A[8][0..3]、A[0][16..19]、A[8][16..19]；接收 lane 1 各块的字节列分别加 4。

B 手工路径从 sBk[k][n] 抽取同一个 n、相邻 k 的字节。ldmatrix 路径使用预先存好的 sBn[n][k]，使相邻 k 连续。两块覆盖 k=0..15、16..31，使用 `.x2`；地址为 `sBn+(lane&7)*32+((lane>>3)&1)*16`。有效地址由 lanes 0..15 提供，其余 lanes 重复合法地址；所有 32 lanes 都执行并接收结果。

接收 lane 0 的 b[0] 含 B[0..3][0]，b[1] 含 B[16..19][0]；lane 1 则含 B[4..7][0]、B[20..23][0]。这正是 fragment_map.cuh 的映射。

这里两条 ldmatrix 均不加 `.trans`。MMA 的 `.row.col` 是操作数解释约定，并不意味着 B 必须用 ldmatrix.trans；后者对 b16 单元进行转置，不能直接等同于 FP8 字节矩阵的转置。shared 数组显式 16-byte 对齐，行跨度 32 字节、子块偏移 0/16 也满足对齐。

`__cvta_generic_to_shared` 将 C++ generic 指针转为 PTX shared 地址；`"=r"` 是输出 b32 寄存器，`"r"` 是输入地址寄存器，`"memory"` 提醒编译器 asm 访问内存；warp 数据就绪由之前的 `__syncwarp()` 保证。

## PTX 静态计数与回答

统计生成 PTX 中 `bar.warp.sync` 之后、`mma.sync` 之前的 smem→fragment 指令。排除 C=0 初始化、D 指针转换、global→shared 准备和结果写回；固定地址立即数偏移不另算一条指令。两条路径均未出现该阶段地址算术被提前移出统计区间的情况；共同的 threadIdx 读取不计。

| 类别 | manual | ldsm |
|---|---:|---:|
| 装载 | 12（4×ld.shared.b32 + 8×ld.shared.b8） | 2（x4 + x2） |
| 地址算术（移位、掩码、OR、加法） | 12 | 10 |
| shared 基址 mov（单列，不算算术） | 2 | 2 |
| 字节打包 | 12（2×mul.wide.u16、4×shl、6×or） | 0 |
| 合计 | 38 | 14 |

若将基址 mov 计入广义地址准备，则分别为 14、12 条。逐条归类见 instruction-counts.json。

(a) ldmatrix 将整行片段按硬件规定直接分发到 warp 各 lane 的目标寄存器，省去大量逐元素 load 与显式打包。本次编译中 A 的 16 次源码 byte 读取已经合并成 4 次 b32 load，不应误报为 24 条手工 load；实际主要额外打包来自 B。

(b) 普通 load 不提供这种矩阵 fragment 分发语义。手工路径必须负责坐标、地址及目标字节顺序。当前 sBk 布局中同一 n 的相邻 k 间隔 8 字节，不能用一次连续 b32 load 得到所需四个字节，因而需要 gather 与打包。并非任何手工写法都必然产生相同数量的指令：调整 B 布局、使用向量读取等可以减少开销，A 的合并本身就是例子。

这里比较的是指定阶段的静态 PTX，不是 SASS 指令数、访存 transaction 数或耗时。B 转为 sBn 的准备工作在统计范围外，不能据此声称整个 kernel 有 38/14 倍加速。

参考：[NVIDIA PTX ldmatrix 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-ldmatrix)。
