# 0.2 理论峰值与机器平衡点

统一口径：单 GPU、dense、FMA=2 FLOP；BF16 输入与 FP32 累加。频率用于标称模型，不把运行时动态时钟或指令依赖延迟当成固定吞吐。TFLOPS、GB/s 均使用十进制单位。2026-09-22 按本人请求由助教查阅官方资料补表。

| 量 | RTX 5090 | B300 SXM |
|---|---|---|
| bf16 FLOP/cycle/SM | 512（依据白皮书峰值、170 SM、2407MHz反算） | 8192（Blackwell tcgen05 满速 BF16 吞吐模型） |
| bf16 峰值(TFLOPS) | 209.5 | 2250 |
| fp8 峰值(TFLOPS) | 位宽估计419；白皮书FP32累加419，FP16累加838；新kind路径需另分口径，见下文 | 位宽估计4500；官方dense4500 |
| fp4 峰值(TFLOPS) | 简单位宽估计838；官方FP32累加dense1676 | 简单位宽估计9000；官方dense13500 |
| datasheet 对照值与口径差异 | BF16 209.5/419、FP8 FP32累加419/838、FP4 1676/3352分别为dense/sparse；不可把3352 AI TOPS当BF16峰值 | HGX为8卡总量：BF16 36PF sparse、FP8 72PF sparse，单卡dense各除以16；FP4明确列144PF sparse/108PF dense，单卡dense除以8 |
| HBM/GDDR 带宽(GB/s) | 1792，GDDR7 | 8000，HBM3e，规格上限 |
| 机器平衡点(FLOP/byte，bf16) | 209.5×1000/1792 ≈ 116.91 | 2250×1000/8000 = 281.25 |

推导采用 `P = SM数 × 每SM每周期FLOP × 频率`。5090 白皮书列170 SM、每SM四个Tensor Core和2407MHz；以512 FLOP/cycle/SM计算为 `170×512×2.407/1000=209.50528 TFLOPS`。这里512来自规格反算，不冒充独立MMA吞吐或延迟微基准；官网2.41GHz是显示精度较粗的值。[RTX Blackwell白皮书，Table 3及脚注](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)

B300 使用8192 FLOP/cycle/SM的BF16模型；项目已留存的[B300设备探测](../team/c1_flashkda/results/environment-20260910T041816895557Z.json)为148 SM。若以官方2250TFLOPS标称值校准，等效频率为 `2250×1000/(148×8192)≈1.85580 GHz`。这是**从官方峰值反推的模型频率，不是读取到的boost规格，也不是独立验证**。取1.86GHz时得到2255.09TFLOPS，与官方舍入值接近。实际实验需另记录动态时钟。CUTLASS说明Blackwell `tcgen05.kind::f16` 相对Hopper吞吐加倍；本表使用相应8192模型，不从一条MMA有多少在飞或延迟多少周期直接推出它。[CUTLASS架构说明](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html)

官方B300数值按HGX B300 SXM口径取值：FP4的dense不是sparse的一半，该表已分别列出108PF/144PF（八卡）。Ultra的dense NVFP4增强使其不同于简单的位宽减半估计。不要把最高配置Blackwell Ultra/GB300宣传中的15PF直接混入本表的B300 SXM 13.5PF。[HGX规格和脚注](https://www.nvidia.com/en-us/data-center/hgx/)、[Blackwell Ultra架构说明](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/)

5090 的 FP8 还必须区分指令路径：白皮书的FP32累加行列419TFLOPS；CUTLASS又明确说明SM120新增 `mma.sync.aligned.kind::f8f6f4` / block-scale路径对FP32 accumulator具有相对Ada FP8的2倍吞吐。不能把419作为所有SM120 FP8指令的统一上限；若后续使用新kind路径，应单独使用其吞吐口径并实测。FP4的官方1676也说明只从BF16按位宽推成四倍会低估新低精度路径。[CUTLASS SM120说明](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html#blackwell-sm120-gemms)

带宽独立核对：[5090官方发布规格](https://www.nvidia.com/en-gb/geforce/news/rtx-50-series-graphics-cards-gpu-laptop-announcements/)、[HGX每GPU带宽表](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory/latest/components.html)。

与单条MMA的3.2 FLOP/byte相比，BF16机器平衡点分别约为其36.5倍和87.9倍。这说明若每次MMA都从显存重新取全部输入，供数无法匹配算力；GEMM需要跨输出复用输入，并通过tiling、shared、寄存器/TMEM与流水提高有效利用率。但3.2是指令操作数口径，不等于整个kernel的实际HBM算术强度，不能直接代入kernel roofline。
