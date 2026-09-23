# B300 cuBLAS tuning: T1, T2, T3 only

User-authorized scope: run T1--T3 from the proposed test list. Do not run T4--T8,
any profiler, the historical nine kernels/pipeline variants, or historical
default-cuBLAS performance measurements again.

| Task | Shapes (M, N, K) | New methods | Cache protocol |
| --- | --- | --- | --- |
| T1 | (4096,4096,4096), (2048,9472,4096), (2048,9728,4096), (2048,9472,8192) | cuBLASLt: request 32 heuristics, 64 MiB workspace, select by actual timing | zero 256 MiB before each timed call, outside events |
| T2 | (2048,9472,4096), (2048,9728,4096) | cuBLAS GemmEx experimental CUBLAS_GEMM_AUTOTUNE | same as T1 |
| T3 | (8192,8192,8192), (12288,12288,12288), (16384,16384,16384) | default torch.mm and tuned cuBLASLt | steady repeated use of the same buffers; no explicit cache flush |

T3's large shapes are new experiments. Default torch.mm is only measured on
these new shapes. A same-buffer steady test does not imply the entire working
set fits in L2. Results from different cache protocols must not be treated as a
controlled speedup comparison.

## Precision and timing

`D = A @ B.T`, FP16 input/output, FP32 accumulate, alpha=1, beta=0. All methods
must disallow FP16 intermediate reduction. cuBLASLt uses a COMPUTE_TYPE
reduction preference mask, accepts only NONE/COMPUTE_TYPE actual reduction
configurations, and inspects numerical implementation flags. GemmEx uses
`CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION`. PyTorch's corresponding flag
is also disabled. TF32 is disabled for the FP32 reference.

Every usable Lt candidate is validated on every output element for seed 0.
Tuning uses three shuffled rounds of ten calls per candidate after ten warmup
calls. Selected methods undergo three-seed validation and a separate five-round
benchmark (up to 150 calls/round, calibrated to about 200 ms). Final outputs are
checked again. The old `atol=0.01, rtol=0.02` criterion is retained.

CUDA events bracket each GEMM; allocation, data generation, reference,
validation, algorithm search, first AUTOTUNE call, warmup, and cache flush are
outside reported latency. Host submission gaps may be included, as in the old
protocol. Formal median latency is the median of round means; raw samples and
round CV are retained. No CUDA Graph or clock/power modification is performed.

32 heuristic candidates are a bounded search, not an exhaustive library search
or a guarantee of the global best implementation. The experimental AUTOTUNE
strategy is a distinct mechanism from cuBLASLt candidate timing.

## Persistence and reproducibility

The native wrapper is compiled on a CPU-only Modal worker before GPU dispatch.
CUDA 13.1 and the existing PyTorch 2.10.0/cu130 + cuBLAS 13.1.0.3 stack are used.
Sources, request, compiler output, binary hashes, actual loaded library paths,
GPU identity and telemetry are recorded. The GPU worker has no automatic retry.

Results are written atomically after each candidate/round and committed to a
durable Modal Volume periodically and at case boundaries. Completed cases,
candidate training rounds, and measurement rounds are skipped on explicit
recovery/resume with the identical request and binary. Library/config changes
must not silently reuse incompatible results.

Historical results are read-only references from the previous runs. Ratios
against those are cross-session comparisons and cannot establish a small
causal improvement. T3 default-vs-tuned comparisons are interleaved within the
same GPU session and cache protocol.

## Official reference implementations

- [cuBLASLt simple autotuning](https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuBLASLt/LtSgemmSimpleAutoTuning)
- [FP16 input/output, FP32 compute example](https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuBLASLt/LtHSHgemmStridedBatchSimple)
- [cuBLAS 13.1 API and precision documentation](https://docs.nvidia.com/cuda/archive/13.1.0/cublas/index.html)
- [NVIDIA HGX hardware specifications](https://www.nvidia.com/en-us/data-center/hgx/):
  B300 dense FP16/BF16 nominal per-GPU denominator = 2250 TFLOPS. This is not a
  measured sustained throughput at the cloud instance's actual clock/power.
