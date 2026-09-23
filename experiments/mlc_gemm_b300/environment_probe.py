"""CPU-only environment preflight; no coursework files are mounted."""
import modal
if modal.is_local():
    from base_image import image
else:
    image = None

app = modal.App("mlc-gemm-b300-environment")

@app.function(image=image, timeout=180, retries=0)
def probe():
    import importlib.metadata
    import inspect
    import subprocess
    import tvm
    from tvm.tirx.bench import bench
    print(subprocess.check_output(["nvcc", "--version"], text=True))
    print({p: importlib.metadata.version(p) for p in
           ("torch", "apache-tvm", "apache-tvm-ffi", "cuda-bindings", "numpy")})
    print(inspect.signature(tvm.compile))
    print(inspect.getsource(bench)[:6000])
    return True

@app.local_entrypoint()
def main():
    probe.remote()
