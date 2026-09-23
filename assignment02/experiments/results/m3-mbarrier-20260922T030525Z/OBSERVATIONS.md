# 3.3 用户 parity 修复验证

2026-09-22，B300 SXM6 AC，CUDA 13.1.80，driver 580.95.05，ARCH=100f。
相对原始实测快照，CUDA 源码唯一变化：mbar_wait(mbar_u32, 0) 改为 mbar_wait(mbar_u32, round & 1)。

rounds={1,2,4} × seed={42,7} 六个用例全部 PASS、exit 0。每例 timeout -k 2s 5s，无重试。
原始记录 ../m3-mbarrier-20260922T021741Z 中 rounds=2/4 的四例均 exit 124。
run.json、inputs 与 PTX 已保存并核对哈希。运行器输出的 Baseline outcomes 是旧标签，本目录实际测试的是上述修复版。

语义解释：正确 phase 等待使本轮 MMA 完成通知先于后续 TMEM load；错误 parity 可能提前确认上一轮完成，使 ld 与未完成的 MMA 发生同址读写冲突，也可能错过本轮完成后等待下一 phase。wait::ld 只等待 load，不能替代 MMA completion 的 mbarrier 等待。原版超时没有 PC 定位，不应声称已确定挂在 ld 或 wait 的具体位置。
