// First correctness-oriented complete C2 partial, not a standalone GEMM.
// One CTA = (query row, KV head, split). GQA16 / D128 / page128.
// K Q^T and V^T P^T both use tcgen05 M128 N16 K128, BF16 -> FP32.
// The first version deliberately uses ordinary thread loads into shared memory;
// it isolates the MMA/TMEM/softmax dataflow before a TMA variant is introduced.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/Float8_e4m3fn.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cutlass/arch/barrier.h>
#include <cmath>
#include <cstdint>

using namespace cute;
using BF16 = cutlass::bfloat16_t;
constexpr int PAGE = 128, DIM = 128, GROUP = 16, THREADS = 128;

struct Params {
  const BF16* q;
  const void* kv;
  const int32_t *indices, *table, *lengths;
  const float *ks, *vs;
  BF16* partial;
  float* lse;
  int rows, kvheads, dql, splits, topk, scale_mode;
  int64_t qs[3], kvs[4], is[3], ts[2], ls;
  int64_t kss[2], vss[2];
  float scale_log2e;
};

using Mma = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_SS<BF16, BF16, float, 128, 16,
                       UMMA::Major::K, UMMA::Major::K>{}));
using AShape = decltype(partition_shape_A(Mma{}, make_shape(Int<128>{}, Int<128>{})));
using BShape = decltype(partition_shape_B(Mma{}, make_shape(Int<16>{}, Int<128>{})));
using ALayout = decltype(UMMA::tile_to_mma_shape(UMMA::Layout_K_SW32_Atom<BF16>{}, AShape{}));
using BLayout = decltype(UMMA::tile_to_mma_shape(UMMA::Layout_K_SW32_Atom<BF16>{}, BShape{}));

struct Shared {
  // A is reused for K and V^T after the preceding MMA is known complete.
  alignas(128) BF16 a[cosize_v<ALayout>];
  alignas(128) BF16 q[cosize_v<BLayout>];
  alignas(128) BF16 prob[cosize_v<BLayout>];
  alignas(128) float mma_out[DIM * GROUP];
  alignas(128) float output[DIM * GROUP];
  float maximum[GROUP], logsumexp[GROUP], alpha[GROUP];
  alignas(16) uint64_t mma_barrier;
  uint32_t tmem;
};

__device__ float warp_max(float x) {
  #pragma unroll
  for (int delta = 16; delta; delta >>= 1) x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, delta));
  return x;
}
__device__ float warp_sum(float x) {
  #pragma unroll
  for (int delta = 16; delta; delta >>= 1) x += __shfl_xor_sync(0xffffffff, x, delta);
  return x;
}

template<bool FP8>
__device__ BF16 load_effective(const Params& p, int page, int head,
                              int token, int dim, bool value) {
  const int64_t at = int64_t(page) * p.kvs[0] + int64_t(head) * p.kvs[1]
      + int64_t(token) * p.kvs[2] + int64_t(dim + (value ? DIM : 0)) * p.kvs[3];
  if constexpr (!FP8) {
    return static_cast<const BF16*>(p.kv)[at];
  } else {
    // Match baseline FP8 -> BF16 -> FP32 scale multiply -> BF16, including
    // physical token/head scale addressing and non-unit backing strides.
    float effective = float(BF16(float(static_cast<const c10::Float8_e4m3fn*>(p.kv)[at])));
    if (p.scale_mode) {
      const float* scale = value ? p.vs : p.ks;
      const int64_t* ss = value ? p.vss : p.kss;
      int64_t offset = p.scale_mode == 1 ? 0 : int64_t(head) * ss[0] + (int64_t(page) * PAGE + token) * ss[1];
      effective *= scale[offset];
    }
    return BF16(effective);
  }
}

template<bool FP8>
__global__ void paged_tcgen05_partial(Params p) {
  int row = blockIdx.x % p.rows, split = blockIdx.x / p.rows;
  int kh = blockIdx.y, tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
  int request = row / p.dql, local_q = row % p.dql;
  int visible = max(0, p.lengths[int64_t(request) * p.ls] - p.dql + local_q + 1);
  int real_topk = min(p.topk, (visible + PAGE - 1) / PAGE);
  int count_per_split = (p.topk + p.splits - 1) / p.splits;
  int first = split * count_per_split;
  int end = min(first + count_per_split, real_topk);
  int heads = p.kvheads * GROUP;
  int64_t part_base = ((int64_t(split) * p.rows + row) * heads + kh * GROUP) * DIM;
  int64_t lse_base = (int64_t(split) * p.rows + row) * heads + kh * GROUP;
  if (first >= end) {
    for (int i = tid; i < GROUP * DIM; i += THREADS) p.partial[part_base + i] = BF16(0.0f);
    if (tid < GROUP) p.lse[lse_base + tid] = -CUDART_INF_F;
    return;
  }

  extern __shared__ __align__(128) char storage[];
  Shared& s = *reinterpret_cast<Shared*>(storage);
  auto mma = Mma{};
  auto thr = mma.get_slice(Int<0>{});
  auto sa = make_tensor(make_smem_ptr(s.a), ALayout{});
  auto sq = make_tensor(make_smem_ptr(s.q), BLayout{});
  auto sp = make_tensor(make_smem_ptr(s.prob), BLayout{});
  auto out = make_tensor(make_smem_ptr(s.mma_out),
                         make_layout(make_shape(Int<128>{}, Int<16>{}), LayoutRight{}));
  auto ca = thr.partition_A(make_identity_tensor(make_shape(Int<128>{}, Int<128>{})));
  auto cb = thr.partition_B(make_identity_tensor(make_shape(Int<16>{}, Int<128>{})));
  for (int i = tid; i < size(sq); i += THREADS) {
    auto ij = cb(i);
    int h = get<0>(ij), d = get<1>(ij);
    sq(i) = p.q[int64_t(row) * p.qs[0] + int64_t(kh * GROUP + h) * p.qs[1] + int64_t(d) * p.qs[2]];
  }
  for (int i = tid; i < DIM * GROUP; i += THREADS) s.output[i] = 0.0f;
  if (tid < GROUP) { s.maximum[tid] = -CUDART_INF_F; s.logsumexp[tid] = -CUDART_INF_F; }
  TMEM::Allocator1Sm allocator;
  if (tid < 32) allocator.allocate(32, &s.tmem);
  if (tid == 0) initialize_barrier(s.mma_barrier, 1);
  cutlass::arch::fence_barrier_init();
  __syncthreads();
  auto ra = thr.make_fragment_A(sa);
  auto rq = thr.make_fragment_B(sq);
  auto rp = thr.make_fragment_B(sp);
  auto co = thr.partition_C(out);
  auto acc = thr.make_fragment_C(co);
  acc.data() = s.tmem;
  auto tc = make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, acc);
  auto copy_thread = tc.get_slice(tid);
  auto dst = copy_thread.partition_D(co);
  auto reg = make_tensor<float>(shape(dst));
  int phase = 0;

  for (int slot = first; slot < end; ++slot) {
    // Only valid slots reach either pointer chase. Ignored sentinel slots are
    // never read, and causal masking uses logical positions, not physical ids.
    int logical = p.indices[int64_t(kh) * p.is[0] + int64_t(row) * p.is[1] + int64_t(slot) * p.is[2]];
    int physical = p.table[int64_t(request) * p.ts[0] + int64_t(logical) * p.ts[1]];
    for (int i = tid; i < size(sa); i += THREADS) {
      auto ij = ca(i);
      int token = get<0>(ij), d = get<1>(ij);
      sa(i) = logical * PAGE + token < visible ? load_effective<FP8>(p, physical, kh, token, d, false) : BF16(0.0f);
    }
    __syncthreads();
    cutlass::arch::fence_view_async_shared();
    mma.accumulate_ = UMMA::ScaleOut::Zero;
    if (tid < 32) {
      #pragma unroll
      for (int kk = 0; kk < DIM / 16; ++kk) {
        gemm(mma, ra(_, _, kk), rq(_, _, kk), acc);
        mma.accumulate_ = UMMA::ScaleOut::One;
      }
      cutlass::arch::umma_arrive(&s.mma_barrier);
      wait_barrier(s.mma_barrier, phase); phase ^= 1;
    }
    __syncthreads();
    copy(tc, copy_thread.partition_S(acc), reg);
    copy(reg, dst);
    __syncthreads();

    // Four warps handle four heads each; token reduction is explicit and does
    // not assume tcgen05.ld.red reduces the transposed token dimension.
    for (int h = warp; h < GROUP; h += 4) {
      float logits[4], local_max = -CUDART_INF_F;
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        int token = lane + 32 * j;
        logits[j] = logical * PAGE + token < visible ? s.mma_out[token * GROUP + h] * p.scale_log2e : -CUDART_INF_F;
        local_max = fmaxf(local_max, logits[j]);
      }
      float new_max = fmaxf(s.maximum[h], warp_max(local_max));
      float local_sum = 0;
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        logits[j] = exp2f(logits[j] - new_max);
        local_sum += logits[j];
      }
      float page_sum = warp_sum(local_sum);
      if (lane == 0) {
        s.alpha[h] = exp2f(s.maximum[h] - new_max);
        s.logsumexp[h] = new_max + log2f(exp2f(s.logsumexp[h] - new_max) + page_sum);
        s.maximum[h] = new_max;
      }
      // Write via the MMA B partition of a logical (head, token) tensor.
      // A shared canonical view is not assumed: cb defines swizzled positions.
      #pragma unroll
      for (int j = 0; j < 4; ++j) s.mma_out[(lane + j * 32) * GROUP + h] = logits[j];
    }
    __syncthreads();
    for (int i = tid; i < size(sp); i += THREADS) {
      auto ij = cb(i); int h = get<0>(ij), token = get<1>(ij);
      sp(i) = BF16(s.mma_out[token * GROUP + h]);
    }
    for (int i = tid; i < size(sa); i += THREADS) {
      auto ij = ca(i); int d = get<0>(ij), token = get<1>(ij);
      sa(i) = logical * PAGE + token < visible ? load_effective<FP8>(p, physical, kh, token, d, true) : BF16(0.0f);
    }
    __syncthreads();
    cutlass::arch::fence_view_async_shared();
    mma.accumulate_ = UMMA::ScaleOut::Zero;
    if (tid < 32) {
      #pragma unroll
      for (int kk = 0; kk < PAGE / 16; ++kk) {
        gemm(mma, ra(_, _, kk), rp(_, _, kk), acc);
        mma.accumulate_ = UMMA::ScaleOut::One;
      }
      cutlass::arch::umma_arrive(&s.mma_barrier);
      wait_barrier(s.mma_barrier, phase); phase ^= 1;
    }
    __syncthreads();
    copy(tc, copy_thread.partition_S(acc), reg);
    copy(reg, dst);
    __syncthreads();
    for (int i = tid; i < DIM * GROUP; i += THREADS) {
      int h = i % GROUP;
      s.output[i] = s.output[i] * s.alpha[h] + s.mma_out[i];
    }
    __syncthreads();
  }
  for (int i = tid; i < GROUP * DIM; i += THREADS) {
    int h = i / DIM, d = i % DIM;
    float normalization = exp2f(s.maximum[h] - s.logsumexp[h]);
    p.partial[part_base + i] = BF16(s.output[d * GROUP + h] * normalization);
  }
  if (tid < GROUP) p.lse[lse_base + tid] = s.logsumexp[tid];
  __syncthreads();
  if (tid < 32) { allocator.release_allocation_lock(); allocator.free(s.tmem, 32); }
}

void partial_cuda(at::Tensor q, at::Tensor kv, at::Tensor topk,
                  at::Tensor table, at::Tensor lengths, at::Tensor ks,
                  at::Tensor vs, at::Tensor partial, at::Tensor lse,
                  int64_t dql, int64_t splits, double sm_scale) {
  TORCH_CHECK(q.is_cuda() && kv.is_cuda() && topk.is_cuda() && table.is_cuda() && lengths.is_cuda(), "CUDA tensors required");
  c10::cuda::CUDAGuard device_guard(q.device());
  TORCH_CHECK(q.scalar_type() == at::kBFloat16 && q.dim() == 3 && q.size(2) == DIM, "Q must be BF16 [rows, heads,128]");
  TORCH_CHECK(kv.dim() == 4 && kv.size(2) == PAGE && kv.size(3) == 2 * DIM && q.size(1) == kv.size(1) * GROUP, "GQA16/page128/D128 required");
  TORCH_CHECK(kv.scalar_type() == at::kBFloat16 || kv.scalar_type() == at::ScalarType::Float8_e4m3fn, "KV must be BF16 or E4M3FN");
  TORCH_CHECK(topk.scalar_type() == at::kInt && table.scalar_type() == at::kInt && lengths.scalar_type() == at::kInt, "indices must be int32");
  TORCH_CHECK(topk.dim() == 3 && topk.size(0) == kv.size(1) && topk.size(1) == q.size(0), "topk shape mismatch");
  TORCH_CHECK(dql > 0 && q.size(0) % dql == 0 && lengths.numel() == q.size(0) / dql, "decode rows mismatch");
  TORCH_CHECK(splits > 0 && splits <= 16 && (splits & (splits - 1)) == 0, "split must be power of two <=16");
  TORCH_CHECK(partial.is_contiguous() && lse.is_contiguous() && partial.scalar_type() == at::kBFloat16 && lse.scalar_type() == at::kFloat, "workspace layout/type mismatch");
  TORCH_CHECK(partial.numel() == splits * q.numel() && lse.numel() == splits * q.size(0) * q.size(1), "workspace size mismatch");
  TORCH_CHECK((ks.numel() == 0) == (vs.numel() == 0), "K/V scales must be supplied together");
  Params p{};
  p.q = reinterpret_cast<const BF16*>(q.data_ptr()); p.kv = kv.data_ptr();
  p.indices = topk.data_ptr<int32_t>(); p.table = table.data_ptr<int32_t>(); p.lengths = lengths.data_ptr<int32_t>();
  p.partial = reinterpret_cast<BF16*>(partial.data_ptr()); p.lse = lse.data_ptr<float>();
  p.rows = q.size(0); p.kvheads = kv.size(1); p.dql = dql; p.splits = splits; p.topk = topk.size(2);
  p.scale_log2e = float(sm_scale) * 1.4426950408889634f;
  for (int i = 0; i < 3; ++i) {p.qs[i] = q.stride(i); p.is[i] = topk.stride(i);}
  for (int i = 0; i < 4; ++i) p.kvs[i] = kv.stride(i);
  p.ts[0] = table.stride(0); p.ts[1] = table.stride(1); p.ls = lengths.stride(0);
  if (ks.numel()) {
    TORCH_CHECK(ks.is_cuda() && vs.is_cuda() && ks.scalar_type() == at::kFloat && vs.scalar_type() == at::kFloat, "scale must be CUDA FP32");
    TORCH_CHECK(ks.sizes() == vs.sizes(), "scale shape mismatch");
    p.ks = ks.data_ptr<float>(); p.vs = vs.data_ptr<float>();
    p.scale_mode = ks.numel() == 1 ? 1 : 2;
    if (p.scale_mode == 2) {
      TORCH_CHECK(ks.dim() == 2 && ks.size(0) == kv.size(1) && ks.size(1) >= kv.size(0) * PAGE, "token/head scale shape mismatch");
      for (int i = 0; i < 2; ++i) {p.kss[i] = ks.stride(i); p.vss[i] = vs.stride(i);}
    }
  }
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(p.rows * p.splits, p.kvheads);
  if (kv.scalar_type() == at::kBFloat16) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(paged_tcgen05_partial<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(Shared)));
    paged_tcgen05_partial<false><<<grid, THREADS, sizeof(Shared), stream>>>(p);
  } else {
    C10_CUDA_CHECK(cudaFuncSetAttribute(paged_tcgen05_partial<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(Shared)));
    paged_tcgen05_partial<true><<<grid, THREADS, sizeof(Shared), stream>>>(p);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
