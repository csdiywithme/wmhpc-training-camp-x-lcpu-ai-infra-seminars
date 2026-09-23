"""Diagnostic recapture of BN128 counters after invalid full-report metrics."""
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import modal
import modal_m4_epilogue_tiles as base
app=modal.App('assignment02-m4-epilogue-counters')

@app.function(image=base.image,gpu='B300',timeout=70,retries=0,max_containers=1,scaledown_window=2)
def run():
    import subprocess
    root=Path('/opt/m4_epi/cuda')
    def cmd(args,timeout=10):
        p=subprocess.run(args,cwd=root,capture_output=True,text=True,timeout=timeout)
        return dict(command=args,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
    path=root/'04h_bn128_counters.ncu-rep'
    metrics='gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__t_requests_pipe_lsu_mem_global_op_st.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,launch__occupancy_limit_shared_mem,sm__cycles_elapsed.avg'
    records=[cmd(['nvidia-smi']),cmd(['timeout','-k','2s','35s','ncu','--clock-control','none','--metrics',metrics,'--kernel-name','regex:gemm_pipeline','--launch-skip','21','--launch-count','1','--export',str(path),'./bin/m4_gemm/04h_bn128','4096','4096','4096'],40)]
    exports={p:cmd(['ncu','--import',str(path),'--page',p]+(['--csv'] if p=='raw' else []),10) for p in ('raw','details')} if path.exists() else {}
    return dict(commands=records,exports=exports,report=path.read_bytes() if path.exists() else b'',source_sha256=hashlib.sha256((root/'m4_gemm/04h_bn128.cu').read_bytes()).hexdigest())

@app.local_entrypoint()
def main():
    out=base.HERE/'results'/('m4-bn128-counters-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir(parents=True)
    source=base.CUDA/'m4_gemm/04h_bn128.cu';(out/source.name).write_bytes(source.read_bytes())
    (out/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    r=run.remote();data=r.pop('report');(out/'04h_bn128_counters.ncu-rep').write_bytes(data)
    r['report_sha256']=hashlib.sha256(data).hexdigest()
    (out/'run.json').write_text(json.dumps(r,indent=2)+'\n')
    for p,v in r['exports'].items():(out/('ncu-'+p+('.csv' if p=='raw' else '.txt'))).write_text(v['stdout'])
    assert hashlib.sha256(source.read_bytes()).hexdigest()==r['source_sha256']
    print('Saved:',out)
    for c in r['commands']: print(c['returncode'],c['stdout'][-500:])
    print(r['exports'].get('details',{}).get('stdout',''))
