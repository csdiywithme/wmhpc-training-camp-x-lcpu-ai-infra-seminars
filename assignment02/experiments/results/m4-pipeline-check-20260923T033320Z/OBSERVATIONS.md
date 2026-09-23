# 4.3 opportunistic prefetch check (S=3)

2026-09-23 on B300 SXM6 AC, CUDA 13.1, nvcc target `sm_100f`. Source SHA256 matches the uploaded and returned snapshot. The code before this edit is [`03_pipeline_before.cu`](03_pipeline_before.cu); the exact edit is [`opportunistic_change.patch`](opportunistic_change.patch); tested code is [`inputs/m4_gemm/03_pipeline.cu`](inputs/m4_gemm/03_pipeline.cu).

Each GPU command used `timeout -k 2s 5s`, with no timeout or automatic retry.

| M × N × K | Result | TFLOPS | cuBLAS TFLOPS |
|---|---|---:|---:|
| 128 × 64 × 64 | PASS, bad=0 | 0.1 | 0.1 |
| 128 × 64 × 128 | PASS, bad=0 | 0.3 | 0.3 |
| 256 × 192 × 256 | PASS, bad=0 | 1.7 | 2.9 |
| 4096³ | PASS, bad=0 | 474.5 | 1762.3 |

The earlier forced-issue version measured 341.5 TFLOPS on 4096³ in a separate B300 allocation. This run improved to 474.5 TFLOPS, but this is not a controlled same-allocation performance comparison and does not complete the 4.3 stage sweep. Full command output: [`run.json`](run.json).
