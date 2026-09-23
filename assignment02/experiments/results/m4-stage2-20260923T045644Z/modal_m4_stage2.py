"""Controlled epilogue and BN sweep, followed by scoped NCU reports."""
import hashlib
import json
import re
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import modal
HERE=Path(__file__).resolve().parent
CUDA=HERE.parent/'cuda'
ROOT=Path('/opt/m4_s2/cuda')
STEMS=('04h_bn128','04i_bn128_s2','04j_bk128_s2')
INPUTS=('common.h','Makefile')+tuple(f'm4_gemm/{s}.cu' for s in STEMS)

def command(argv,timeout=10):
    try:
        p=subprocess.run(argv,cwd=ROOT,capture_output=True,text=True,timeout=timeout)
        r=dict(command=argv,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
    except subprocess.TimeoutExpired as e:
        def decode(s): return s.decode(errors='replace') if isinstance(s,bytes) else (s or '')
        r=dict(command=argv,returncode=None,stdout=decode(e.stdout),stderr=decode(e.stderr),error='timeout')
    print(json.dumps(r),flush=True)
    return r

def build():
    records=[]
    for stem in STEMS:
        r=command(['make','-B','ARCH=100f','FLAGS=-O2 -std=c++17 -I. --expt-relaxed-constexpr -lineinfo -DSTAGES='+('3' if stem=='04h_bn128' else '2'),f'bin/m4_gemm/{stem}'],120)
        records.append(r)
        if r['returncode'] != 0: raise RuntimeError(r)
    (ROOT/'build.json').write_text(json.dumps(records,indent=2))

image=modal.Image.from_registry('nvidia/cuda:13.1.0-devel-ubuntu24.04',add_python='3.11').entrypoint([]).apt_install('build-essential','cuda-nsight-compute-13-1')
for name in INPUTS: image=image.add_local_file(CUDA/name,str(ROOT/name),copy=True)
image=image.run_function(build,timeout=300)
app=modal.App('assignment02-m4-stage2')

@app.function(image=image,gpu='B300',timeout=240,retries=0,max_containers=1,scaledown_window=2)
def run():
    records=[command(['nvidia-smi']),command(['ncu','--version'])]
    valid=[]
    for stem in STEMS:
        bn=128
        bk=128 if stem=='04j_bk128_s2' else 64
        shapes=((128,bn,bk),(128,bn,2*bk),(256,3*bn,5*bk),(256,4096,16384))
        for shape in shapes:
            r=command(['timeout','-k','2s','5s',f'./bin/m4_gemm/{stem}',*map(str,shape)])
            records.append(r)
            if r['returncode'] != 0 or 'PASS(bad=0)' not in r['stdout']: break
        else: valid.append(stem)
    timing={s:[] for s in valid}
    # Rotate order across rounds to reduce fixed position bias.
    for i in range(3):
        for stem in valid[i:]+valid[:i]:
            r=command(['timeout','-k','2s','5s',f'./bin/m4_gemm/{stem}','4096','4096','4096'])
            records.append(r)
            if r['returncode']!=0 or 'PASS(bad=0)' not in r['stdout']:
                raise RuntimeError('Benchmark correctness failure')
            timing[stem].append(float(re.search(r'([\d.]+) TFLOPS',r['stdout'])[1]))
    medians={s:statistics.median(v) for s,v in timing.items()}
    winner=max(medians,key=medians.get)
    print('MEDIANS '+json.dumps(medians)+' WINNER '+winner,flush=True)
    reports={}; exports={}
    for stem in valid:
        path=ROOT/f'{stem}_4096.ncu-rep'
        r=command(['timeout','-k','2s','35s','ncu','--clock-control','none','--metrics','gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__t_requests_pipe_lsu_mem_global_op_st.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,launch__occupancy_limit_shared_mem,sm__cycles_elapsed.avg','--kernel-name','regex:gemm_pipeline','--launch-skip','21','--launch-count','1','--import-source','yes','--source-folders',str(ROOT),'--export',str(path),f'./bin/m4_gemm/{stem}','4096','4096','4096'],40)
        records.append(r)
        if path.exists():
            reports[stem]=path.read_bytes(); exports[stem]={}
            for page in ('details','raw'):
                args=['ncu','--import',str(path),'--page',page]
                if page=='raw': args+=['--csv']
                exports[stem][page]=command(args,15)
    return dict(commands=records,medians=medians,winner=winner,reports=reports,exports=exports,build=json.loads((ROOT/'build.json').read_text()),input_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in INPUTS})

@app.local_entrypoint()
def main():
    out=HERE/'results'/('m4-stage2-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir(parents=True)
    for name in INPUTS:
        p=out/'inputs'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((CUDA/name).read_bytes())
    (out/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print('OUTPUT:',out,flush=True)
    try:
        result=run.remote(); reports=result.pop('reports');result['report_sha256']={}
        for stem,data in reports.items():
            (out/f'{stem}_4096.ncu-rep').write_bytes(data)
            result['report_sha256'][stem]=hashlib.sha256(data).hexdigest()
        for stem,pages in result['exports'].items():
            for page,r in pages.items():
                (out/f'{stem}-ncu-{page}{".csv" if page=="raw" else ".txt"}').write_text(r['stdout'])
        (out/'run.json').write_text(json.dumps(result,indent=2)+'\n')
        for n in INPUTS: assert hashlib.sha256((out/'inputs'/n).read_bytes()).hexdigest()==result['input_sha256'][n]
        print('Saved:',out, 'Medians:',result['medians'])
        if any(r['returncode']!=0 for r in result['commands']):raise RuntimeError('Some commands failed; see run.json')
    except Exception as e:
        (out/'error.txt').write_text(str(e)+'\n');raise
