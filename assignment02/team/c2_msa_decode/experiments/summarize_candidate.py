"""Build the final C2 candidate evidence report from immutable saved results."""
import ast
from collections import defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
PAIR=ROOT/"results/candidate-paired-20260913T022747Z"
VERIFY=ROOT/"results/candidate-verify-20260913T022150Z"
CAL=ROOT/"results/candidate-calibrate-20260913T020759Z"
PROFILE=ROOT/"results/candidate-profile-20260913T022426Z"

def read(path):
    return json.loads(path.read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    paired=read(PAIR/"paired.json");verification=read(VERIFY/"candidate-heldout.json")
    calibration=read(CAL/"calibration.json")
    assert len(paired)==64 and verification["status"]=="PASS"
    assert len(verification["baseline_records"])==len(verification["candidate_records"])==168
    assert len(calibration["records"])==168 and calibration["status"]=="CALIBRATION_FROZEN"
    assert sha(ROOT/"candidate.py")==sha(PAIR/"sources/candidate.py")==sha(VERIFY/"sources/candidate.py")
    for r in paired:
        assert len(r["orders"])==21
        assert all(v["finite"] and v["nrmse"]<.02 for v in r["post_graph_correctness"].values())
    # Evidence that the chosen partial is the pinned algorithm, not a rewritten
    # near-equivalent one hidden behind a baseline label.
    source=ast.parse((ROOT/"vllm_msa_ref/sparse_attn.py").read_text())
    target=ast.parse((ROOT/"candidate.py").read_text())
    original=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=="_gqa_sparse_decode_kernel")
    copied=next(n for n in target.body if isinstance(n,ast.FunctionDef) and n.name=="_page_decode_kernel")
    copied.name=original.name
    assert ast.dump(original,include_attributes=False)==ast.dump(copied,include_attributes=False)
    groups=defaultdict(list)
    for row in paired:
        groups[row["pdl"],row["tp"],row["batch"],row["dtype"]].append(row)
    tables={False:[],True:[]};aggregate=[]
    for (pdl,tp,batch,dtype),rows in sorted(groups.items()):
        assert len(rows)==2
        a=statistics.mean(r["baseline"]["median_us"] for r in rows)
        b=statistics.mean(r["candidate"]["median_us"] for r in rows)
        lo,hi=min(r["speedup"] for r in rows),max(r["speedup"] for r in rows)
        tables[pdl].append(f"|{tp}|{batch}|{dtype}|{a:.3f}|{b:.3f}|{a/b:.3f}×|{lo:.3f}–{hi:.3f}×|")
        aggregate.append(dict(tp=tp,batch=batch,dtype=dtype,pdl=pdl,baseline_us=a,candidate_us=b,speedup=a/b,seed_min_speedup=lo,seed_max_speedup=hi))
    text=(PROFILE/"selected.csv").read_text();reader=list(csv.DictReader(io.StringIO(text[text.index('"ID"'):])));units=reader[0]
    ncu=[]
    for r in reader:
        if not r["ID"].isdigit():continue
        ncu.append({k:r.get(k) for k in ("ID","Kernel Name","gpu__time_duration.sum","launch__grid_size",
                   "launch__registers_per_thread","memory_l1_wavefronts_shared","memory_l1_wavefronts_shared_ideal",
                   "derived__memory_l1_wavefronts_shared_excessive","derived__local_spilling_requests")})
    assert len(ncu)==4 and all(float(r["derived__memory_l1_wavefronts_shared_excessive"])==0 for r in ncu)
    no_pdl=[r["speedup"] for r in paired if not r["pdl"]]
    pdl=[r["speedup"] for r in paired if r["pdl"]]
    report=f'''# C2 候选实现、验收与最终性能

## 固定的最终实现

`candidate.run(case)` 默认使用原始 partial 与新的 feature-tiled merge。原 partial 在文件中仅重命名为 `_page_decode_kernel`；AST 比较含参数、装饰器与函数体完全一致。默认保留原 split 策略、4 warps/3 stages、BF16 概率与 partial、FP8 scale 舍入、两 kernel 结构。

merge 改为 1 warp：S≤8 时每 CTA 写128个输出通道；S16时写64个通道。接口还保留实验用 subpage partial，但最终默认不选择它。策略根据 seed0 探索数据固定，之后用 seeds101/307 独立复测，未根据复测负结果偷偷改变策略。

实现 SHA256：`{sha(ROOT/'candidate.py')}`。

## 探索结果保留

- 首轮小形状15配置：`results/candidate-smoke-20260913T021041Z`，全部正确；该轮TP4 B1 BF16无正收益。
- 全矩阵240配置：`results/candidate-tune-20260913T021243Z`，全部完成；细分token tile有些FP8配置变快，BF16多数不利，没有据此宣称普遍加速。
- 保留原partial、仅改merge的192配置：`results/candidate-merge-20260913T021620Z`，全部完成；据此选择简单的split相关merge策略。
- 早期64-token候选通过过full验收，但该PASS仅属于相应源码快照。最终merge-only再次独立通过同一冻结manifest。

## 完整正确性验收

baseline calibration：168条记录，状态`CALIBRATION_FROZEN`，协议digest `{calibration['protocol_digest']}`。
最终heldout：baseline168条、candidate168条，均`PASS`。覆盖三类KV存储、TP1/TP4、B1/4/8/16、物理重排、非恒定scale、stride、tail、dql2/4、空split和padding等冻结域。

候选最大绝对误差 {max(r['metrics']['max_abs'] for r in verification['candidate_records']):.9f}；最大逐行NRMSE {max(r['metrics']['row_nrmse_max'] for r in verification['candidate_records']):.9f}。这些不是全模型质量指标，也不是所有可能输入的误差上界。

最终paired性能任务还对每行baseline/candidate的实际graph重放后输出做FP64检查：64组均finite且NRMSE<0.02。该补充检查位于计时区间之外，覆盖PDL开/关下实际被测graph；不能替代full数值协议的其他边界。

## 最终测量协议

- 同一B300任务内，相同Q/KV/页表、相同PDL设置、分别预分配workspace，比较完整partial+merge链。
- 每图64次调用，预热后测21个repeat；每repeat随机选择baseline/candidate先后顺序，记录完整顺序和原始样本。
- 所有形状使用独立performance seeds101/307；表格耗时为两个seed各自21样本中位数的算术平均。速度比=同行baseline/candidate，最后列显示两个seed的速度比范围。
- CUDA Graph重复固定地址，无主动清缓存，不等于所有数据都在L2；不包含Python wrapper分配/提交、QKV投影、indexer或服务排队。
- 单卡未锁频；结论是本机型与该协议的证据。21个样本是重复graph平均值，P05/P95不是服务p95，也不被称为统计置信区间。

## PDL=false：对应课程独立harness

|TP|Batch|KV|baseline μs|candidate μs|速度比|两个seed范围|
|---|---|---|---|---|---|---|
{chr(10).join(tables[False])}

32个seed/形状结果的速度比范围 **{min(no_pdl):.3f}–{max(no_pdl):.3f}×**，几何平均 **{statistics.geometric_mean(no_pdl):.3f}×**。这对应约{(1-1/min(no_pdl))*100:.1f}%–{(1-1/max(no_pdl))*100:.1f}%延迟降低，不能把速度比减一直接当作延迟降低比例。

## PDL=true：生产相关补充

|TP|Batch|KV|baseline μs|candidate μs|速度比|两个seed范围|
|---|---|---|---|---|---|---|
{chr(10).join(tables[True])}

速度比范围 **{min(pdl):.3f}–{max(pdl):.3f}×**。TP1 B1及TP4 B4存在退化，不能无条件替换生产PDL路径。该结论来自同卡、相同PDL条件；没有用跨任务时钟差异解释掉负结果。
线程块数、merge资源和PDL重叠会共同影响chain；未采集这些退化形状的完整PDL时间线，故不把某一种调度解释写成定论。默认非PDL实验已显示局部优化值得做，但生产推广需要额外dispatch/PDL策略实验。

## NCU对改动机制的验证

单独的cold NCU使用`cache-control=all`、`clock-control=none`、detailed22passes/kernel。TP1 BF16：

|Batch|merge版本|grid CTA|registers/thread|SMEM actual/ideal wavefront|cold duration μs|
|---|---|---|---|---|---|
|1|原始|64|32|69888/8448|5.216|
|1|候选|128|39|640/640|4.672|
|16|原始|1024|22|577536/86016|6.720|
|16|候选|1024|32|0/0|5.152|

候选B1/B16 merge动态shared分别128/0 bytes，两个case的excess shared wavefront和local spill请求均0。原partial的工作量与资源保持不变。此证据支持减少跨warp布局/归约共享内存开销是改进机制；没有将所有时间差唯一归因于某一个counter。
上述原始/候选NCU来自不同任务；它们支持工作量变化与机制解释，正式速度比以同卡paired表为准。cold NCU时间也不与hot graph表混算。

## 复现与原始证据

```bash
uv --cache-dir /private/tmp/a02-uv-cache run --offline --no-project --with modal==1.5.5 python -m modal run assignment02/team/c2_msa_decode/modal_candidate.py --mode calibrate
uv --cache-dir /private/tmp/a02-uv-cache run --offline --no-project --with modal==1.5.5 python -m modal run assignment02/team/c2_msa_decode/modal_candidate.py --mode verify --manifest assignment02/team/c2_msa_decode/results/candidate-calibrate-20260913T020759Z/calibration.json
uv --cache-dir /private/tmp/a02-uv-cache run --offline --no-project --with modal==1.5.5 python -m modal run assignment02/team/c2_msa_decode/modal_candidate.py --mode paired
uv --cache-dir /private/tmp/a02-uv-cache run --offline --no-project --with modal==1.5.5 python -m modal run assignment02/team/c2_msa_decode/modal_candidate.py --mode profile
python3 assignment02/team/c2_msa_decode/experiments/summarize_candidate.py
```

新校准生成新时间戳路径；运行新verify时替换manifest路径，不覆盖旧manifest。

- [最终实现](../candidate.py)
- [设计与先测后做记录](CANDIDATE_DESIGN.md)
- [验收协议](../validation/ACCEPTANCE.md) / [内部独立审查](../validation/INTERNAL_REVIEW.md)
- [冻结manifest](../{CAL.relative_to(ROOT)}/calibration.json)
- [最终heldout验收](../{VERIFY.relative_to(ROOT)}/candidate-heldout.json)
- [最终paired原始样本](../{PAIR.relative_to(ROOT)}/paired.json)
- [候选NCU CSV](../{PROFILE.relative_to(ROOT)}/selected.csv)

每个Modal任务保存上传源码快照、命令与GPU环境。失败和探索任务未删除；正式结论只引用这里明确标出的成功最终结果。
'''
    (ROOT/"experiments/CANDIDATE_EXPERIMENTS.md").write_text(report)
    (ROOT/"experiments/candidate_summary.json").write_text(json.dumps(dict(
        candidate_sha256=sha(ROOT/"candidate.py"),paired_source=str(PAIR.relative_to(ROOT)),
        validation_status=verification["status"],protocol_digest=calibration["protocol_digest"],
        aggregate=aggregate,ncu=ncu),indent=2)+"\n")
    print('Verified source identity, 64 graph comparisons, 168+168 heldout records; wrote candidate report')

if __name__=="__main__":main()
