"""Regenerate the C2 TMA microexperiment report from persisted raw JSON."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
folder = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/paged-copy-20260913T015626Z"
raw = json.loads((folder / "result.json").read_text())
run = next(r for r in raw["records"] if r["command"] == ["/opt/paged_copy"])
assert run["returncode"] == 0, run
records = [json.loads(line) for line in run["stdout"].splitlines() if line.startswith("{")]
env, data = records[0], records[1:]
assert len(data) == 32 and all(r["mismatches"] == 0 for r in data)
grouped = {}
for row in data:
    key = row["heads"], row["batch"], row["element_bytes"]
    grouped.setdefault(key, {})[row["path"]] = row
lines = []
for (heads, batch, size), paths in sorted(grouped.items()):
    sm, tma = paths["sm_vector"]["median_us"], paths["tma"]["median_us"]
    lines.append(f"|{4 // heads}|{batch}|{size}|{sm:.3f}|{tma:.3f}|{sm / tma:.3f}×|")
report = f'''# C2 分页 TMA 搬运微实验

## 结论与证据范围

在 {env['gpu']}（CC {env['cc']}、{env['sms']} SM）上，32 个配置全部逐字节对拍通过。
实际二进制包含 `UTMALDG.4D`；TMA kernel 中仍然保留两次普通 `LDG.E` 来取得逻辑块和物理页号。
因此，**两级间接寻址由 SM 完成，得到物理页后可用规则 tensor map 进行页内搬运**。

这个实验是 C2 讨论点 3 的文档结论验证，不是候选 attention kernel，也不代表完整 decode 加速。
1 字节和 2 字节只对应 FP8/BF16 的存储字节数；测试搬运原始位，不执行浮点反量化。

## 方法

- 编译：CUDA 13.1，`nvcc -O3 -lineinfo -arch=sm_103a`，源代码哈希 `{raw['source_sha256']}`。
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
{chr(10).join(lines)}

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
- [原始环境、命令与样本](../results/{folder.name}/result.json)
- [实际 SASS](../results/{folder.name}/paged_copy.sass)
- [NVIDIA TMA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)
- [PTX cp.async.bulk.tensor](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor)
'''
(ROOT / "experiments/TMA_EXPERIMENT.md").write_text(report)
(folder / "measurements.json").write_text(json.dumps(data, indent=2) + "\n")
print(f"Wrote TMA_EXPERIMENT.md; {len(data)} bitwise checks passed")
