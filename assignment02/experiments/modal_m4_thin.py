"""Assignment 4.5: original 63-point cuBLAS sweep, bounded B300 run."""
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import modal

HERE = Path(__file__).resolve().parent
CUDA = HERE.parent / 'cuda'
ROOT = Path('/opt/m4_thin/cuda')
INPUTS = ('common.h', 'Makefile', 'm4_gemm/05_thin_gemm.cu')
SHAPES = [(1536,128,'f_b_proj'),(2304,1536,'q_b_proj'),(7168,1536,'o_proj'),(2112,7168,'fused_qkv_a_proj'),(6288,7168,'in_proj_qkvgfab'),(7168,8448,'dense_down_proj'),(16896,7168,'dense_gate_up_proj')]
MS = (1,8,16,64,256,1024,4096,16384,65536)

def command(argv, timeout=10):
    p = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    return dict(command=argv, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)

def build():
    r = command(['make','-B','ARCH=100f','bin/m4_gemm/05_thin_gemm'],120)
    (ROOT/'build.json').write_text(json.dumps(r,indent=2))
    if r['returncode']: raise RuntimeError(r)

image = modal.Image.from_registry('nvidia/cuda:13.1.0-devel-ubuntu24.04',add_python='3.11').entrypoint([]).apt_install('build-essential')
for name in INPUTS:
    image = image.add_local_file(CUDA/name,str(ROOT/name),copy=True)
image = image.run_function(build,timeout=180)
app = modal.App('assignment02-m4-thin')

@app.function(image=image,gpu='B300',memory=4096,timeout=90,retries=0,max_containers=1,scaledown_window=2)
def run():
    records = [command(['nvidia-smi']),command(['nvcc','--version'])]
    records.append(command(['timeout','-k','2s','45s','stdbuf','-oL','./bin/m4_gemm/05_thin_gemm','2250','8000'],50))
    records.append(command(['nvidia-smi']))
    return dict(commands=records,build=json.loads((ROOT/'build.json').read_text()),input_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in INPUTS})

@app.local_entrypoint()
def main():
    out = HERE/'results'/('m4-thin-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir(parents=True)
    for name in INPUTS:
        p = out/'inputs'/name
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes((CUDA/name).read_bytes())
    (out/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    with (out/'predictions.csv').open('w') as f:
        w = csv.writer(f)
        w.writerow(['layer','M','N','K','AI_FLOP_per_byte','compute_roof_TFLOPS','memory_roof_TFLOPS','roof_TFLOPS','predicted_bound'])
        for n,k,name in SHAPES:
            for m in MS:
                ai = m*n*k/(m*k+n*k+m*n)
                w.writerow([name,m,n,k,ai,2250,ai*8,min(2250,ai*8),'memory' if ai<281.25 else 'compute'])
    print('Predictions saved before GPU run:',out,flush=True)
    try:
        result = run.remote()
        (out/'run.json').write_text(json.dumps(result,indent=2)+'\n')
        (out/'stdout.txt').write_text(result['commands'][2]['stdout'])
        for name in INPUTS:
            assert hashlib.sha256((out/'inputs'/name).read_bytes()).hexdigest()==result['input_sha256'][name]
        print(result['commands'][2]['stdout'])
        print('Saved:',out)
        if any(r['returncode'] for r in result['commands']): raise RuntimeError('Command failed; see run.json')
    except Exception as e:
        (out/'error.txt').write_text(str(e)+'\n')
        raise
