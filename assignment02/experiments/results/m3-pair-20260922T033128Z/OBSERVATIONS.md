# 3.4 CTA pair B300 实验

2026-09-22，按用户请求运行给定 04_cta_pair.cu；没有修改 kernel。环境 B300 SXM6 AC，CC10.3，driver580.95.05，nvcc13.1.80，ARCH100f。实际源码在 inputs，运行元数据在 run.json，PTX 和 NCU 原始输出在本目录。

## 正确性与容量

两种实现均 PASS，bad=0。每 block shared：24588 B / 20492 B（含12B管理空间）；纯 A/B 为24/20KiB。每 CTA 的 A 均16KiB，B为8/4KiB，TMEM均32KiB。全grid操作数staging为48/40KiB。

## NCU

使用 --clock-control none --csv --launch-count 2，仅采集两种实现各自首个 kernel，各有两个 block。指标先查询再采集；详细名称见 summary.json。

| 指标 | group1 | group2 |
|---|---:|---:|
| sm__sass_data_bytes_mem_shared_op_st.sum |49162|40970|
| l1tex__data_pipe_tc_wavefronts_mem_shared.sum |384|320|
| A operand wavefront |256|256|
| B operand wavefront（分别1cta、2cta scope）|128|64|
| LSU shared store wavefront |778|652|

B读取工作量减半，A不变，总Tensor Core shared wavefront为5/6。Store字节数比纯A/B各多10B，包括管理写入；两实现之差8192B。不能将不同层级计数器相加或把所有wavefront无条件换算成相同字节数。

普通运行13.22/14.32us；按题意不据此判断快慢。NCU重放打印的144.12/152.93us也不作为性能结果。

## 超时和结果

correctness：timeout -k 2s 5s；NCU：timeout -k 2s 35s。GPU函数上限120秒，无自动重试。两项正常完成，测后compute-apps为空。

Modal app：https://modal.com/apps/simidawhu/main/ap-CUkZzaBJCxwuTqvy09t1Uf

3.4(c/d)尚未讨论，本次不代填。完整整理见 handout/src/assignment02.md 的个人记录。
