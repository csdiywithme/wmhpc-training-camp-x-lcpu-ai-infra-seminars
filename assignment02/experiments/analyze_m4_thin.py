"""Join the handout program's printed measurements with pre-run roofs."""
import csv
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
r = json.loads((out/'run.json').read_text())
assert all(c['returncode'] == 0 for c in r['commands'])
rows = list(csv.DictReader((out/'predictions.csv').open()))
measurements = {}
for line in (out/'stdout.txt').read_text().splitlines():
    p = line.split()
    if len(p) != 10 or p[0] == 'layer': continue
    key = (p[0],p[1],p[2],p[3])
    assert key not in measurements
    measurements[key] = dict(us=p[4],TFLOPS=p[5],effective_GB_per_s=p[6],printed_AI=p[7],compute_attainment_pct=p[8].rstrip('%'),memory_attainment_pct=p[9].rstrip('%'))
assert len(measurements) == len(rows) == 63
for row in rows:
    row.update(measurements[(row['layer'],row['M'],row['N'],row['K'])])
    row['roof_attainment_pct'] = 100*float(row['TFLOPS'])/float(row['roof_TFLOPS'])
with (out/'measurements.csv').open('w') as f:
    w = csv.DictWriter(f,fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
lines = ['# 4.5 B300 瘦 GEMM：63 点记录','','理论值取 0.2：2250 TFLOPS、8000 GB/s，平衡点 281.25 FLOP/byte。AI 和 roof 已在 GPU 调用前保存于 predictions.csv。实测由原始程序输出（保留一位小数）；完整合并数据见 measurements.csv。','','CUDA 13.1；原程序、common.h、Makefile 与 runner 快照均在本目录。每个形状先单次调用并同步，再 warmup 20 次，CUDA event 计时 200/50/20 次平均。整轮程序 timeout 45s、kill grace 2s，无重试。','','注意：有效 GB/s 是 2(MK+NK+MN)/time，不是 NCU 测量的 HBM 流量；重复使用权重可能命中缓存。此程序没有输出数值对拍，也没有检查 cuBLAS 返回状态，完成计时不等于数值正确性验证。未运行 skinny CUDA Core 对照，不能声称验证题设的 8%–100% 收益。','','| layer | M | N | K | AI | compute roof TF/s | memory roof TF/s | time µs | TF/s | compute % | memory % |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for v in rows:
    lines.append(f"| {v['layer']} | {v['M']} | {v['N']} | {v['K']} | {float(v['AI_FLOP_per_byte']):.2f} | 2250 | {float(v['memory_roof_TFLOPS']):.2f} | {v['us']} | {v['TFLOPS']} | {v['compute_attainment_pct']} | {v['memory_attainment_pct']} |")
(out/'OBSERVATIONS.md').write_text('\n'.join(lines)+'\n')
print('Validated and joined all 63 points:',out)
for layer in dict.fromkeys(v['layer'] for v in rows):
    print(layer,[(v['M'],v['us'],v['TFLOPS'],v['compute_attainment_pct'],v['memory_attainment_pct']) for v in rows if v['layer']==layer])
