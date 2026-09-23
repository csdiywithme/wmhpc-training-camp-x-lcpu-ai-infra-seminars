"""Step 1: one short B300 CUDA/NCU probe; no FlashKDA benchmark yet.

Run locally: python -m modal run assignment02/team/c1_flashkda/modal_probe.py
The CUDA binary is compiled during the CPU-only image build.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import modal


CUDA_SOURCE = r"""
#include <cuda_runtime.h>
#include <cstdio>
#define CHECK(expr) do { cudaError_t e = (expr); if (e != cudaSuccess) { \
  std::fprintf(stderr, "%s: %s\n", #expr, cudaGetErrorString(e)); return 1; } } while (0)
__global__ void probe(int* x) { x[threadIdx.x] = threadIdx.x + 1; }
int main() {
  cudaDeviceProp p;
  CHECK(cudaGetDeviceProperties(&p, 0));
  std::printf("GPU=%s CC=%d.%d SMs=%d\n", p.name, p.major, p.minor, p.multiProcessorCount);
  int* d; int h[32];
  CHECK(cudaMalloc(&d, sizeof(h)));
  probe<<<1, 32>>>(d);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(d));
  for (int i = 0; i < 32; ++i) if (h[i] != i + 1) return 2;
  std::puts("CUDA_SMOKE_PASS");
}
"""

image = (
    modal.Image.from_registry("nvidia/cuda:13.1.0-devel-ubuntu24.04", add_python="3.11")
    .entrypoint([])
    .apt_install("cuda-nsight-compute-13-1")
    .run_commands(
        "cat > /tmp/probe.cu <<'CUDA_EOF'\n" + CUDA_SOURCE + "\nCUDA_EOF",
        "nvcc -O2 -lineinfo -gencode arch=compute_103a,code=sm_103a /tmp/probe.cu -o /opt/c1-probe",
    )
)
app = modal.App("c1-environment-probe")


@app.function(
    image=image, gpu="B300", cpu=1, memory=2048,
    timeout=180, retries=0, max_containers=1, scaledown_window=2,
)
def probe_environment():
    import glob
    import shutil
    import subprocess

    results = []

    def run(argv, timeout=30):
        try:
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
            item = dict(command=argv, returncode=proc.returncode,
                        stdout=proc.stdout, stderr=proc.stderr)
        except (OSError, subprocess.TimeoutExpired) as exc:
            item = dict(command=argv, returncode=None, error=str(exc))
        results.append(item)
        return item

    run(["nvidia-smi"])
    run(["nvcc", "--version"])
    # Read only the profiling-related driver setting, if exposed by the host.
    try:
        params = Path("/proc/driver/nvidia/params").read_text()
        results.append(dict(driver_profiling_settings=[
            line for line in params.splitlines() if "Profil" in line
        ]))
    except OSError as exc:
        results.append(dict(driver_profiling_settings_error=str(exc)))
    smoke = run(["/opt/c1-probe"])
    candidates = sorted(glob.glob("/opt/nvidia/nsight-compute/*/ncu"))
    ncu = shutil.which("ncu") or (candidates[-1] if candidates else None)
    if ncu:
        run([ncu, "--version"])
        if smoke["returncode"] == 0:
            # Diagnostic only: avoid clock changes, cache flushing, and kernel replay.
            # These settings are not a prescription for the final benchmark methodology.
            run([ncu, "--clock-control", "none", "--cache-control", "none",
                 "--replay-mode", "application", "--metrics", "sm__cycles_elapsed.avg",
                 "--launch-count", "1", "/opt/c1-probe"], timeout=60)
    else:
        results.append(dict(error="NCU executable not found; counters remain unverified."))
    return results


@app.local_entrypoint()
def main():
    results = probe_environment.remote()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(__file__).resolve().parent / "results" / f"environment-{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    for result in results:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Saved: {output}")
