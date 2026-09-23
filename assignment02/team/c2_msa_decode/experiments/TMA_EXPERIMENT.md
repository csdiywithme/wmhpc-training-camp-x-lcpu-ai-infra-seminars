# C2 分页 TMA 搬运微实验

## 结论与证据范围

在 NVIDIA B300 SXM6 AC（CC 10.3、148 SM）上，32 个配置全部逐字节对拍通过。
实际二进制包含 `UTMALDG.4D`；TMA kernel 中仍然保留两次普通 `LDG.E` 来取得逻辑块和物理页号。
因此，**两级间接寻址由 SM 完成，得到物理页后可用规则 tensor map 进行页内搬运**。

这个实验是 C2 讨论点 3 的文档结论验证，不是候选 attention kernel，也不代表完整 decode 加速。
1 字节和 2 字节只对应 FP8/BF16 的存储字节数；测试搬运原始位，不执行浮点反量化。

## 方法

- 编译：CUDA 13.1，`nvcc -O3 -lineinfo -arch=sm_103a`，源代码哈希 `c8820f27c0f2c295a87fe3960fbf8a7043ee0ea46aafa6071b2a1a7a527f8768`。
- 布局与 MSA 一致：`[physical_page, KV_head, 128, 256]`，最后一维是拼接 K/V。
- 每个请求有 64 个逻辑页，选择 16 个不同逻辑页；物理页打乱，每个 head 独立选择。
- 两个 kernel 均使用 128 线程和一页 shared buffer，均执行 global → shared → global；输出布局与校验完全相同。
- 线程路径使用 128 位向量 load/store；TMA 路径使用 4D tensor map、32/64 KiB transaction 和 mbarrier。
- TMA map 的 box 为 `[256,128,1,1]`，采用无 swizzle；动态坐标为 `[0,0,head,physical_page]`。
- 同一 kernel 预热后捕获 100 个节点的 CUDA Graph，测量 9 次 replay 的每次调用平均耗时；表内为 9 个样本的中位数。
- 地址重复、没有主动清缓存。大工作集可能超过 L2；这里的预热不保证所有输入都驻留 L2。
- 先测线程路径再测 TMA，没有随机交错执行。因此这是一轮机制验证与条件性能样本，不将差值外推为普遍上限。
- CPU 按原始 top-k 与 page table 独立构造期望输出；逐字节比较，而非与另一个 GPU kernel 互相当作 gold。

## 原始结果汇总

单位 μs；TP1 = 4 个 KV heads，TP4 = 1 个 KV head；存储列单位 byte/element。

|TP|Batch|存储|线程向量搬运|TMA 搬运|线程/TMA|
|---|---|---|---|---|---|
|4|1|1|2.827|2.376|1.190×|
|4|1|2|4.117|3.035|1.357×|
|4|4|1|2.867|2.398|1.196×|
|4|4|2|4.181|3.120|1.340×|
|4|8|1|2.943|2.522|1.167×|
|4|8|2|4.241|3.344|1.268×|
|4|16|1|3.544|3.239|1.094×|
|4|16|2|5.491|4.580|1.199×|
|1|1|1|2.868|2.396|1.197×|
|1|1|2|4.189|3.128|1.339×|
|1|4|1|3.542|3.178|1.114×|
|1|4|2|5.489|4.567|1.202×|
|1|8|1|4.996|4.380|1.141×|
|1|8|2|12.227|7.607|1.607×|
|1|16|1|9.610|7.004|1.372×|
|1|16|2|31.299|20.535|1.524×|

## 限制与实际 attention 集成要求

这里每个 CTA 只搬一个完整页面，不计算 QK、softmax、PV，也不做流水线重叠。
完整 attention 的布局、shared 占用、反量化、Tensor Core 供数和 CTA 数量均可能改变收益。
没有验证 `cp.async` 路径，也没有声称线程向量路径是最快的非 TMA 实现。
数据量列是逻辑读写量，不能当作 NCU 测得的 HBM 流量。

测试使用完整有效页面，不覆盖 causal tail、空请求或无效 top-k slot。
真实集成必须在页表查询前判断 slot 有效性；对已分配尾页中的未来 token 单独做逻辑 mask。
TMA 的物理 OOB 行为不能替代这些语义；也不能保护先发生的非法 `block_table` 访问。

## 复现与产物

```bash
uv --cache-dir /private/tmp/a02-uv-cache run --offline --no-project --with modal==1.5.5 python -m modal run assignment02/team/c2_msa_decode/modal_copy.py
python3 assignment02/team/c2_msa_decode/experiments/summarize_copy.py
```

- [CUDA 源码](paged_copy.cu)
- [Modal 运行器](../modal_copy.py)
- [原始环境、命令与样本](../results/paged-copy-20260913T015626Z/result.json)
- [实际 SASS](../results/paged-copy-20260913T015626Z/paged_copy.sass)
- [NVIDIA TMA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)
- [PTX cp.async.bulk.tensor](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor)
