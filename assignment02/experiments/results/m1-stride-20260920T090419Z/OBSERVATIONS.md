# 1.5 B300 stride 实验

2026-09-20，用户先给出预测 8/16/32/4，再授权助教运行。给定 05_ldsm_stride.cu 未修改，真实源码及 SHA256 核对结果保存在 inputs/ 与 run.json。

环境：simidawhu / B300 SXM6 AC (CC10.3)，driver580.95.05，CUDA13.1.80，ARCH100f，NCU2025.4.1.0。Modal app https://modal.com/apps/simidawhu/main/ap-eO5UjsOhhuTzYqXMoEiknf 已结束。

## 结果

| stride | 单次 kernel wavefront总数 | conflict总数 | 每条实际warp LDSM wavefront | 每条实际warp LDSM conflict | 原程序打印cycle |
|---|---:|---:|---:|---:|---:|
|32B|16384|8192|8|4|9.72|
|64B|32768|24576|16|12|10.75|
|128B|65536|57344|32|28|16.06|
|144B|8192|0|4|0|9.23|

NCU 同时采集每档 warmup 和正式 kernel，各档两次计数完全一致；表中为一个 kernel，不累加两次。三次独立普通运行在程序输出的两位小数精度下完全一致。计时取自普通运行，不采用 NCU replay 时打印值。原始数据见 ncu-output.txt、timing-output.txt、run.json；结构化归一化见 summary.json。

## 必须保留的编译器现象

按源码 8*4096=32768 次 warp LDSM 直接归一化是错误的。PTX 中循环展开32次，但最终 cuobjdump SASS 每个循环体只剩两条 LDSM.16.M88.4（各函数偏移0x510与0x530）。循环计数加0x20，终止值0x1000，因此循环128次，每warp实际256条LDSM，全kernel共2048条。

这解释了绝对计数比源码预期少16倍：8*256*{8,16,32,4} 正好对应实测 wavefront。

数据地址恒定、无中途写入，重复装载在 PTX→SASS 阶段被合并。asm volatile 保留了 PTX 中的指令，并没有保证最终机器码保留每次源码装载。最终 SASS 见 05_ldsm_stride.sass。

## 结论与计时解释

用户的bank模型预测被实测支持：wavefront比2:4:8:1，128B相对32B是4倍，padding冲突为0。

原程序cycle计算是thread0的(t1-t0)/4096，没有再除8，也不是单条LDSM延迟。128B/32B原始计时比16.06/9.72约1.65。循环中还保留大量XOR对应的LOP3指令、循环控制与相关依赖，且有8个warp交错执行；wavefront代表shared服务工作量，不等于整个循环周期。这里尤其不能忽略装载合并的影响，不能仅凭这两个计数器声称已经测到纯LSU瓶颈或量化了延迟隐藏的贡献。

若要测量更纯粹的持续LDSM吞吐，需要另做防装载合并的对照版本并检查SASS。本次保留题目原样的结果，不擅自把源码改成另一个基准。

## 操作记录

首次运行 m1-stride-20260920T090243Z 普通计时成功，NCU因无法锁GPU时钟返回9；没有修改主机时钟。第二次运行 m1-stride-20260920T090327Z 使用 --clock-control none，采集成功。第三次即本目录，补采SASS并复核计数与普通计时。

运行器：../../modal_m1_stride.py，上传仅common.h、Makefile、05_ldsm_stride.cu及运行器。执行命令为 MODAL_PROFILE=simidawhu uv --cache-dir /private/tmp/a02-uv-cache run --offline --no-project --with 'modal[api-proxy-support]==1.5.5' python -m modal run assignment02/experiments/modal_m1_stride.py。
