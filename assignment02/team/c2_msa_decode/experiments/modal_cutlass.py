"""Fixed public upstream MSA/CUTLASS comparison; one bounded GPU per run."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GPU = os.environ.get("C2_GPU", "B300")
MSA_PIN = "087c161814d4d9c735b46c21212a09e5f8eb92fa"
CUTLASS_PIN = "eb61c911471867a5fd2466bfd8f29306cea6ebf8"
WARM_CACHE = os.environ.get("C2_CUTLASS_WARM_CACHE")
image = (modal.Image.from_registry("nvidia/cuda:13.1.0-devel-ubuntu24.04", add_python="3.11").entrypoint([])
         .apt_install("git", "cuda-nsight-compute-13-1")
         .pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cu130")
         .pip_install("numpy==2.2.6")
         .apt_install("curl")
         .pip_install("apache-tvm-ffi==0.1.13.post3", "ninja==1.13.0", "pybind11==3.0.1")
         .run_commands(
             "mkdir -p /opt/msa /opt/provenance",
             f"curl -fLsS https://codeload.github.com/vllm-project/MSA/tar.gz/{MSA_PIN} -o /opt/provenance/msa.tar.gz",
             "tar -xzf /opt/provenance/msa.tar.gz -C /opt/msa --strip-components=1",
             f"curl -fLsS https://codeload.github.com/NVIDIA/cutlass/tar.gz/{CUTLASS_PIN} -o /opt/provenance/cutlass.tar.gz",
             "tar -xzf /opt/provenance/cutlass.tar.gz -C /opt/msa/python/fmha_sm100/cutlass --strip-components=1",
             "sha256sum /opt/provenance/*.tar.gz > /opt/provenance/sha256.txt")
         .env({"MINFER_FMHA_CACHE_DIR":"/tmp/c2-cutlass/compile-cache", "MAX_JOBS":"4"}))
if WARM_CACHE:
    # Reuse only the four compiled public-upstream libraries. Generated code,
    # compiler flags and full raw outputs remain in the earlier result record.
    cache_root = Path(WARM_CACHE).resolve()
    for library in sorted(cache_root.glob("*/*.so")):
        image = image.add_local_file(library, f"/tmp/c2-cutlass/compile-cache/{library.relative_to(cache_root)}")
for filename in ("harness/synth.py", "harness/vllm_shim.py", "vllm_msa_ref/sparse_attn.py",
                 "vllm_msa_ref/msa_cutlass_sparse_decode.py", "experiments/baseline_workload.py",
                 "experiments/cutlass_workload.py"):
    image = image.add_local_file(ROOT / filename, f"/opt/c2/{filename}")
app = modal.App("c2-msa-upstream-cutlass")


@app.function(image=image, gpu=GPU, cpu=4, memory=24576, timeout=600,
              retries=0, max_containers=1, scaledown_window=2)
def run(batches, tps, pdl, controls):
    import subprocess
    import time
    output = Path("/tmp/c2-cutlass")
    output.mkdir(exist_ok=True)
    records=[]
    def command(label, argv, timeout=30):
        begin=time.monotonic()
        try:
            p=subprocess.run(argv,capture_output=True,text=True,timeout=timeout)
            row=dict(label=label,argv=argv,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
        except subprocess.TimeoutExpired as exc:
            row=dict(label=label,argv=argv,returncode=None,error="timeout",
                     stdout=exc.stdout.decode(errors="replace") if isinstance(exc.stdout,bytes) else exc.stdout,
                     stderr=exc.stderr.decode(errors="replace") if isinstance(exc.stderr,bytes) else exc.stderr)
        row["elapsed_seconds"]=time.monotonic()-begin
        records.append(row)
        (output/"commands.json").write_text(json.dumps(records,indent=2))
        print(json.dumps(row),flush=True)
    command("nvidia-smi",["nvidia-smi"])
    command("pip-freeze",["python","-m","pip","freeze"])
    command("nvcc",["nvcc","--version"])
    command("archive-sha256",["cat","/opt/provenance/sha256.txt"])
    command("cutlass",["python","-u","/opt/c2/experiments/cutlass_workload.py",
                       "--output",str(output),"--batches",batches,"--tps",tps]+(["--pdl"] if pdl else [])+(["--controls"] if controls else []),480)
    for library in output.glob("compile-cache/**/*.so"):
        p=subprocess.run(["cuobjdump","--dump-resource-usage",str(library)],capture_output=True,text=True,timeout=15)
        library.with_suffix(".resources.txt").write_text(p.stdout+p.stderr)
        p=subprocess.run(["cuobjdump","--dump-sass",str(library)],capture_output=True,text=True,timeout=20)
        library.with_suffix(".sass").write_text(p.stdout+p.stderr)
    command("nvidia-smi-after",["nvidia-smi"])
    return dict(msa_pin=MSA_PIN,cutlass_pin=CUTLASS_PIN,gpu_requested=GPU,
                commands=records,artifacts={str(p.relative_to(output)):p.read_bytes()
                                           for p in output.rglob("*") if p.is_file()})


@app.local_entrypoint()
def main(batches:str="16,32,64",tps:str="1,4",pdl:bool=True,controls:bool=False):
    output=HERE/"results"/f"cutlass-{GPU.lower()}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    output.mkdir(parents=True,exist_ok=False)
    for name in ("modal_cutlass.py","cutlass_workload.py","baseline_workload.py"):
        (output/name).write_bytes((HERE/name).read_bytes())
    print(f"C2_OUTPUT={output}",flush=True)
    try:
        result=run.remote(batches,tps,pdl,controls)
    except Exception as exc:
        (output/"error.json").write_text(json.dumps(dict(error=repr(exc)),indent=2))
        raise
    for name,content in result.pop("artifacts").items():
        target=output/name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(content)
    result["warm_cache_source"] = WARM_CACHE
    (output/"run.json").write_text(json.dumps(result,indent=2))
    print(f"C2_SAVED={output}",flush=True)
    if any(row["returncode"]!=0 for row in result["commands"]):
        raise RuntimeError("Upstream CUTLASS comparison failed; raw artifacts saved")
