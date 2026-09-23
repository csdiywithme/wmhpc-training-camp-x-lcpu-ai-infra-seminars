"""C1 C16/32/64 range, FP16 Neumann, and cancellation GPU microbench.

Run: python chunk_numeric_gpu.py --output /tmp/c1-artifacts/chunk-numeric.json
This isolates numerical mechanisms. Its dense Neumann implementation writes
intermediates to global memory and is NOT FlashKDA's fused register-only K1.
The custom GEMM explicitly uses SM80 m16n8k16 F16-input/F16-accumulator MMA.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess

import torch
from torch.utils.cpp_extension import load_inline


CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

__device__ __forceinline__ unsigned pack(__half x, __half y) {
    return static_cast<unsigned>(__half_as_ushort(x)) |
           (static_cast<unsigned>(__half_as_ushort(y)) << 16);
}

__device__ __forceinline__ void atom(unsigned &d0, unsigned &d1,
    unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                 "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
                 : "+r"(d0), "+r"(d1)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Each warp computes a 16x16 C tile as two m16n8k16 atoms along N.
// Explicit PTX lane layouts; no CUTLASS source or installed FlashKDA dependency.
__global__ void half_gemm(const __half* A, const __half* B, __half* C, int n) {
    int lane=threadIdx.x%32, warp=threadIdx.x/32;
    int tiles=n/16, tile=blockIdx.x*4+warp;
    if(tile>=tiles*tiles) return;
    int row0=(tile/tiles)*16, col0=(tile%tiles)*16;
    int g=lane/4, t=lane%4;
    unsigned d0=0, d1=0, d2=0, d3=0;
    for(int k0=0;k0<n;k0+=16) {
        unsigned a0=*reinterpret_cast<const unsigned*>(A+(row0+g)*n+k0+2*t);
        unsigned a1=*reinterpret_cast<const unsigned*>(A+(row0+g+8)*n+k0+2*t);
        unsigned a2=*reinterpret_cast<const unsigned*>(A+(row0+g)*n+k0+2*t+8);
        unsigned a3=*reinterpret_cast<const unsigned*>(A+(row0+g+8)*n+k0+2*t+8);
        unsigned b0=pack(B[(k0+2*t)*n+col0+g], B[(k0+2*t+1)*n+col0+g]);
        unsigned b1=pack(B[(k0+2*t+8)*n+col0+g], B[(k0+2*t+9)*n+col0+g]);
        unsigned b2=pack(B[(k0+2*t)*n+col0+g+8], B[(k0+2*t+1)*n+col0+g+8]);
        unsigned b3=pack(B[(k0+2*t+8)*n+col0+g+8], B[(k0+2*t+9)*n+col0+g+8]);
        atom(d0,d1,a0,a1,a2,a3,b0,b1);
        atom(d2,d3,a0,a1,a2,a3,b2,b3);
    }
    *reinterpret_cast<unsigned*>(C+(row0+g)*n+col0+2*t)=d0;
    *reinterpret_cast<unsigned*>(C+(row0+g+8)*n+col0+2*t)=d1;
    *reinterpret_cast<unsigned*>(C+(row0+g)*n+col0+2*t+8)=d2;
    *reinterpret_cast<unsigned*>(C+(row0+g+8)*n+col0+2*t+8)=d3;
}

torch::Tensor mma_half(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda() && a.device()==b.device(), "CUDA inputs required");
    TORCH_CHECK(a.scalar_type()==torch::kFloat16 && b.scalar_type()==torch::kFloat16,
                "FP16 inputs required");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && a.dim()==2 && b.dim()==2,
                "contiguous matrices required");
    int n=a.size(0);
    TORCH_CHECK(a.size(1)==n && b.size(0)==n && b.size(1)==n && n%16==0,
                "square matrices with n multiple of 16 required");
    c10::cuda::CUDAGuard guard(a.device());
    auto c=torch::empty_like(a);
    int tiles=(n/16)*(n/16);
    half_gemm<<<(tiles+3)/4,128,0,c10::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(a.data_ptr()),
        reinterpret_cast<const __half*>(b.data_ptr()), reinterpret_cast<__half*>(c.data_ptr()), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return c;
}

__global__ void exp_factors(const float* g, float* ef, float* invf,
                            __nv_bfloat16* eb, __nv_bfloat16* ib, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    float x=g[i]*1.4426950408889634f, e, inv;
    asm("ex2.approx.ftz.f32 %0,%1;" : "=f"(e) : "f"(x));
    asm("ex2.approx.ftz.f32 %0,%1;" : "=f"(inv) : "f"(-x));
    ef[i]=e; invf[i]=inv;
    eb[i]=__float2bfloat16_rn(e); ib[i]=__float2bfloat16_rn(inv);
}

std::vector<torch::Tensor> factors(torch::Tensor g) {
    TORCH_CHECK(g.is_cuda() && g.scalar_type()==torch::kFloat32 && g.is_contiguous(),
                "contiguous CUDA FP32 exponents required");
    c10::cuda::CUDAGuard guard(g.device());
    auto ef=torch::empty_like(g), invf=torch::empty_like(g);
    auto eb=torch::empty_like(g,g.options().dtype(torch::kBFloat16));
    auto ib=torch::empty_like(eb);
    exp_factors<<<(g.numel()+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(
        g.data_ptr<float>(), ef.data_ptr<float>(), invf.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(eb.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(ib.data_ptr()), g.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {ef,invf,eb,ib};
}
"""


def build_extension():
    return load_inline(
        name="c1_chunk_numeric_sm80_v1",
        cpp_sources=["torch::Tensor mma_half(torch::Tensor a, torch::Tensor b);",
                     "std::vector<torch::Tensor> factors(torch::Tensor g);"],
        cuda_sources=CUDA_SOURCE,
        functions=["mma_half", "factors"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
        verbose=True,
    )


def tensor_stats(tensor):
    x = tensor.detach().double()
    finite = torch.isfinite(x)
    return {
        "dtype": str(tensor.dtype), "elements": x.numel(),
        "nan": int(torch.isnan(x).sum().item()), "inf": int(torch.isinf(x).sum().item()),
        "zeros": int((x == 0).sum().item()),
        "finite_abs_max": x[finite].abs().max().item() if bool(finite.any()) else None,
    }


def errors(actual, expected):
    if not bool(torch.isfinite(actual).all()):
        return {"max_abs": None, "relative_l2": None}
    delta = actual.double() - expected.double()
    return {"max_abs": delta.abs().max().item(),
            "relative_l2": (delta.norm() / expected.double().norm().clamp_min(1e-30)).item()}


def validate_mma(extension):
    records = []
    torch.manual_seed(711)
    for n in (16, 32, 64):
        identity = torch.eye(n, device="cuda", dtype=torch.float16)
        coded = (torch.arange(n * n, device="cuda").reshape(n, n) % 127).half() / 128
        left = extension.mma_half(identity, coded)
        right = extension.mma_half(coded, identity)
        exact = bool(torch.equal(left, coded) and torch.equal(right, coded))
        a = (torch.randn(n, n, device="cuda") * 0.1).half()
        b = (torch.randn(n, n, device="cuda") * 0.1).half()
        got = extension.mma_half(a, b)
        delta = errors(got, a.double() @ b.double())
        passed = exact and delta["max_abs"] is not None and delta["max_abs"] < 0.005
        records.append({"n": n, "identity_exact": exact, "random_error": delta, "pass": passed})
        if not passed:
            raise RuntimeError(f"Custom MMA lane-layout/numeric smoke failed: {records[-1]}")
    return records


def neumann(matrix, multiply, stages=False):
    size = matrix.shape[0]
    identity = torch.eye(size, device=matrix.device, dtype=matrix.dtype)
    inverse, power = identity - matrix, matrix
    trace = []
    for level in range(1, int(math.log2(size))):
        power = multiply(power, power)
        if stages:
            trace.append({"stage": f"power_{2 ** level}", **tensor_stats(power)})
        inverse = inverse + multiply(inverse, power)
        if stages:
            trace.append({"stage": f"partial_inverse_{2 ** (level + 1)}", **tensor_stats(inverse)})
    return inverse, trace


def graph_time(function, repetitions=50):
    for _ in range(3):
        function()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = function()
    samples = []
    for _ in range(7):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repetitions):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / repetitions)
    # Keep captured output alive until all replays finish.
    del result
    return {"median_us": statistics.median(samples), "samples_us": samples,
            "scope": "CUDA Graph replay of dense global-intermediate doubling; not fused K1 latency"}


def range_experiments(extension):
    records = []
    for n in (16, 32, 64):
        for per_token in (-5.0, -0.01):
            g = torch.arange(1, n + 1, device="cuda", dtype=torch.float32) * per_token
            ef, inf, eb, ib = extension.factors(g)
            causal = torch.tril(torch.ones(n, n, device="cuda", dtype=torch.bool))
            factored = eb[:, None] * ib[None, :]
            expected = torch.exp(g.double()[:, None] - g.double()[None, :])
            # Exact v1 factorization for k=1, versus exponentiating a difference.
            restored = ib * eb[-1]
            direct_restored = extension.factors((g[-1] - g).contiguous())[2]
            center = g[-1] / 2  # Include G0=0 in the interval being centered.
            centered = extension.factors((g - center).contiguous())
            zero_positions = torch.nonzero(ef == 0).flatten()
            infinite_positions = torch.nonzero(torch.isinf(ib)).flatten()
            records.append({
                "chunk": n, "log_gate_per_token": per_token,
                "exact_exp_min": math.exp(n * per_token), "exact_inverse_max": math.exp(-n * per_token),
                "exp_fp32": tensor_stats(ef), "inverse_fp32": tensor_stats(inf),
                "exp_bf16": tensor_stats(eb), "inverse_bf16": tensor_stats(ib),
                "first_zero_exp_token_1based": int(zero_positions[0].item()) + 1 if zero_positions.numel() else None,
                "first_infinite_inverse_token_1based": int(infinite_positions[0].item()) + 1 if infinite_positions.numel() else None,
                "causal_factored_pairs": tensor_stats(factored[causal]),
                "causal_pair_error": errors(factored[causal], expected[causal]),
                "restored_via_product": tensor_stats(restored),
                "restored_via_difference": tensor_stats(direct_restored),
                "restored_product_error": errors(restored, torch.exp((g[-1] - g).double())),
                "centered_exp_bf16": tensor_stats(centered[2]),
                "centered_inverse_bf16": tensor_stats(centered[3]),
                "centering_scope": "range of factors only; not an implementation of rescaled KDA",
            })
    return records


def inverse_experiments(extension, with_timings=True):
    records = []
    torch.manual_seed(1207)
    for n in (16, 32, 64):
        row = torch.arange(n, device="cuda", dtype=torch.float64)
        distance = row[:, None] - row[None, :]
        cases = []
        for beta in (1.0, 0.990234375, 0.75):
            cases.append((f"aligned_gate0_beta{beta}", torch.tril(torch.full((n, n), beta,
                           device="cuda", dtype=torch.float64), diagonal=-1)))
        cases.append(("aligned_decay0.1_beta1", torch.where(distance > 0, torch.exp(-0.1 * distance), 0)))
        keys = torch.randn(n, 128, device="cuda", dtype=torch.float64)
        keys /= keys.norm(dim=-1, keepdim=True)
        cases.append(("random_normalized_keys_gate0_beta0.7", torch.tril(0.7 * (keys @ keys.T), diagonal=-1)))
        for name, unrounded in cases:
            low = unrounded.half().contiguous()
            # Reference is the inverse of the represented L supplied to the
            # algorithm; input FP16 rounding error is reported separately.
            represented = low.double()
            ident = torch.eye(n, device="cuda", dtype=torch.float64)
            gold = torch.linalg.solve_triangular(ident + represented, ident, upper=False, unitriangular=True)
            modes = (
                ("explicit_sm80_fp16_acc", low, extension.mma_half),
                ("torch_fp16_output_fp32_acc", low, torch.mm),
                ("torch_fp32_tf32_off", low.float(), torch.mm),
            )
            for mode, matrix, multiply in modes:
                output, trace = neumann(matrix, multiply, stages=True)
                residual = (ident + represented) @ output.double() - ident
                first_bad = next((stage["stage"] for stage in trace if stage["nan"] or stage["inf"]), None)
                record = {
                    "chunk": n, "case": name, "mode": mode,
                    "input_rounding": errors(represented, unrounded),
                    "dense_gemms": 2 * (int(math.log2(n)) - 1),
                    "dense_flops": 4 * (int(math.log2(n)) - 1) * n ** 3,
                    "stages": trace, "first_nonfinite_stage": first_bad,
                    "result": tensor_stats(output), "gold": tensor_stats(gold),
                    "inverse_error": errors(output, gold),
                    "residual_abs_max": residual.abs().max().item() if bool(torch.isfinite(residual).all()) else None,
                    "corner": output[-1, 0].item() if bool(torch.isfinite(output[-1, 0])) else None,
                    "gold_corner": gold[-1, 0].item(),
                }
                if with_timings and name == "random_normalized_keys_gate0_beta0.7":
                    record["timing"] = graph_time(lambda: neumann(matrix, multiply)[0])
                records.append(record)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/tmp/c1-artifacts/chunk-numeric.json"))
    parser.add_argument("--no-timings", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new result path")
    if not torch.cuda.is_available():
        parser.error("CUDA GPU required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    if hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
        torch.backends.cuda.matmul.allow_fp16_accumulation = False
    extension = build_extension()
    result = {
        "scope": "GPU microbench of isolated range/Neumann mechanisms; no FlashKDA end-to-end claim",
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cuda_source_sha256": hashlib.sha256(CUDA_SOURCE.encode()).hexdigest(),
        "extension": str(extension.__file__),
        "mma_validation": validate_mma(extension),
        "range": range_experiments(extension),
        "inverse": inverse_experiments(extension, not args.no_timings),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cuobjdump = shutil.which("cuobjdump")
    if cuobjdump:
        dump = subprocess.run([cuobjdump, "--dump-sass", str(extension.__file__)],
                              capture_output=True, text=True, check=False)
        sass_path = args.output.with_suffix(".sass")
        sass_path.write_text(dump.stdout)
        result["sass"] = {"path": str(sass_path), "returncode": dump.returncode,
                          "stderr": dump.stderr, "scope": "inspect actual HMMA type before attributing precision"}
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"result": str(args.output), "mma_validation": result["mma_validation"],
                      "range_cases": len(result["range"]), "inverse_cases": len(result["inverse"])}))


if __name__ == "__main__":
    main()
