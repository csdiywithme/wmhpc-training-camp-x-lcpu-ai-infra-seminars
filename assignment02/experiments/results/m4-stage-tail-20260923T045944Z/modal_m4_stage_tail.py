"""S2/S3 comparison on a shape with exactly divisible persistent tile work."""
import hashlib,json,statistics,re
from pathlib import Path
from datetime import datetime,timezone
import modal
import modal_m4_stage2 as base
app=modal.App('assignment02-m4-stage-tail')
@app.function(image=base.image,gpu='B300',timeout=70,retries=0,max_containers=1,scaledown_window=2)
def run():
    import subprocess
    root=Path('/opt/m4_s2/cuda')
    def cmd(args):
        p=subprocess.run(args,cwd=root,capture_output=True,text=True,timeout=10)
        r=dict(command=args,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
        print(json.dumps(r),flush=True);return r
    stems=('04h_bn128','04i_bn128_s2');records=[cmd(['nvidia-smi'])];times={s:[] for s in stems}
    for i in range(3):
        for s in (stems if i%2==0 else stems[::-1]):
            r=cmd(['timeout','-k','2s','5s',f'./bin/m4_gemm/{s}','3072','4736','4096']);records.append(r)
            if r['returncode']!=0 or 'PASS(bad=0)' not in r['stdout']:break
            times[s].append(float(re.search(r'([\d.]+) TFLOPS',r['stdout'])[1]))
        else:continue
        break
    return dict(commands=records,times=times,medians={s:statistics.median(v) for s,v in times.items() if v},source_sha256={s:hashlib.sha256((root/f'm4_gemm/{s}.cu').read_bytes()).hexdigest() for s in stems})
@app.local_entrypoint()
def main():
    out=base.HERE/'results'/('m4-stage-tail-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'));out.mkdir(parents=True)
    for s in ('04h_bn128','04i_bn128_s2'):(out/(s+'.cu')).write_bytes((base.CUDA/f'm4_gemm/{s}.cu').read_bytes())
    (out/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    r=run.remote();(out/'run.json').write_text(json.dumps(r,indent=2)+'\n')
    for s,h in r['source_sha256'].items():assert hashlib.sha256((out/(s+'.cu')).read_bytes()).hexdigest()==h
    assert all(x['returncode']==0 and ('PASS(bad=0)' in x['stdout'] or x['command']==['nvidia-smi']) for x in r['commands'])
    print('Saved:',out,'Medians:',r['medians'])
