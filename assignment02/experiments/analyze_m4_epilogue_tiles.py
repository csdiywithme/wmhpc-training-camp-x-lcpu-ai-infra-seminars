"""Generate measured comparison tables, preserving NCU as separate evidence."""
import csv, hashlib, json, re, sys
from pathlib import Path
out=Path(sys.argv[1]);r=json.loads((out/'run.json').read_text())
for name,digest in r['input_sha256'].items():
    assert hashlib.sha256((out/'inputs'/name).read_bytes()).hexdigest()==digest
for stem,digest in r['report_sha256'].items():
    assert hashlib.sha256((out/f'{stem}_4096.ncu-rep').read_bytes()).hexdigest()==digest
bench={};checks={}
for c in r['commands']:
    a=c['command']
    if len(a)>5 and a[4].startswith('./bin/'):
        stem=Path(a[4]).name
        checks.setdefault(stem,[]).append((a[-3:],c['returncode'],'PASS(bad=0)' in c['stdout']))
        if a[-3:]==['4096']*3:
            bench.setdefault(stem,[]).append(float(re.search(r'([\d.]+) TFLOPS',c['stdout'])[1]))
assert all(code==0 and passed for cs in checks.values() for _,code,passed in cs)
base=r['medians']['04ab_warp_persistent']
lines=['# 4.4 追加实验：epilogue 与 tile 大小','','同次 Modal B300 分配，四版均用 CUDA 13.1、sm_100f、-O2、-lineinfo、STAGES=3。每版四个范围明确的形状严格对拍，再在 4096³ 轮换顺序测三轮。普通程序 5s timeout + 2s grace，NCU 35s + 2s，无自动重试，未锁频。每个程序内包含预热和多次 CUDA event 平均。','','本轮 04g 包含 shared scratch 复用前的 `fence.proxy.async.shared::cta`；较早的 04g 初测快照未包含它，初测仅保留作历史，不作为最终同步实现。','','| 版本 | 三轮 TFLOPS | 中位数 | 对原 AB | 理论 2250 TFLOPS 达成率 |','|---|---|---:|---:|---:|']
for stem,med in r['medians'].items():
    lines.append(f'| {stem} | '+', '.join(map(str,bench[stem]))+f' | {med:.1f} | {(med/base-1)*100:+.1f}% | {med/2250*100:.1f}% |')
lines+=['','推荐版本：`'+r['winner']+'`（仅以本轮 4096³ 结果选择）。','','## 验证范围','']
for stem,cs in checks.items():
    lines.append('- '+stem+': '+', '.join('×'.join(v) for v,_,_ in cs[:4])+'，以及三次 4096³，全部 PASS(bad=0)。')
lines+=['','输入为固定 seed 的 BF16 小整数、FP32 累加/输出，对 cuBLAS FP32 输出逐元素严格比较；没有覆盖随机实数、非 tile 整除形状或所有 GPU。BN=128/256 版本分别要求 N 按对应 BN 整除。','','## NCU（与普通计时分开）','','`--set full --clock-control none --launch-skip 21 --launch-count 1`。原生报告、details/raw/source 导出均在本目录；版本、原始命令及源码/报告 hash 见 run.json。','','| 版本 | duration µs | SM throughput % | store requests | store sectors | sectors/request | shared 限制 CTA/SM |','|---|---:|---:|---:|---:|---:|---:|']
metrics={}
for stem in r['report_sha256']:
    rows=list(csv.DictReader((out/f'{stem}-ncu-raw.csv').open()))
    row=rows[-1]
    keys=['gpu__time_duration.sum','sm__throughput.avg.pct_of_peak_sustained_elapsed','l1tex__t_requests_pipe_lsu_mem_global_op_st.sum','l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum','launch__occupancy_limit_shared_mem']
    vals=[float(row[k].replace(',','')) for k in keys]
    metrics[stem]=dict(zip(keys,vals))
    dur,sm,req,sect,ctas=vals
    if '(!) nan' in (out/f'{stem}-ncu-details.txt').read_text():
        lines.append(f'| {stem} | {dur:.2f} | 无效 | 无效 | 无效 | 无效 | {ctas:.0f} |')
    else:
        lines.append(f'| {stem} | {dur:.2f} | {sm:.2f} | {req:.0f} | {sect:.0f} | {sect/req:.1f} | {ctas:.0f} |')
lines+=['','NCU store sectors 是 L1/TEX 请求的 32B sector 数，不是实际 HBM 写出字节。相同 FP32 输出大小下，sectors/request 降低说明合并写入改善；不能把这些计数直接当成 HBM 流量节省。NCU replay 下主程序打印的 TFLOPS 不参与上面的性能比较。']
(out/'OBSERVATIONS.md').write_text('\n'.join(lines)+'\n')
(out/'selected_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
print('\n'.join(lines))
