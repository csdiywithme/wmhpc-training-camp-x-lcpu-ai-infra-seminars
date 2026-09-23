# 4.3 intermediate pipeline check (S=3)

2026-09-23 on NVIDIA B300 SXM6 AC. The uploaded source snapshot matches the remote SHA256 records in `run.json`. CUDA 13.1 compiled `03_pipeline.cu` with `ARCH=100f` and `STAGES=3` successfully. Each correctness command used `timeout -k 2s 5s`; none timed out.

| M × N × K | Result | Kernel TFLOPS | cuBLAS TFLOPS |
|---|---|---:|---:|
| 128 × 64 × 64 | PASS, bad=0 | 0.1 | 0.1 |
| 128 × 64 × 128 | PASS, bad=0 | 0.3 | 0.3 |
| 256 × 192 × 256 | PASS, bad=0 | 1.8 | 3.4 |
| 4096³ | PASS, bad=0 | 341.5 | 1767.9 |

This is an intermediate forced-issue pipeline. It warms all stages, waits for the previous MMA before reusing a stage, and drains the last MMA before reading TMEM. It has no opportunistic deeper prefetch. The 4096³ timing was 0.40 ms, below the earlier 4.2 TMA observation of 523.5 TFLOPS. This does not establish the final 4.3 stage performance; further overlap and controlled comparisons remain.

See `run.json` for command outputs and `inputs/` for the exact tested source. The comparison checks the program's exact `bad=0` criterion against cuBLAS; it is not a claim about arbitrary shapes or sanitizer coverage.
