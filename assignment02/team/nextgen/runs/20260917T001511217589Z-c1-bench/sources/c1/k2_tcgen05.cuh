// Experimental complete C16 KDA recurrence, not an isolated GEMM benchmark.
// Source lineage: FlashKDA 1ce47ea K2 numerical boundaries; CUTLASS SM100
// tutorial primitives as exercised by ../tile_microbench.cu. Upstream is unmodified.
#pragma once
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/algorithm/cooperative_copy.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/bfloat16.h>
#include <stdexcept>
#include <string>

namespace c1_nextgen {
using namespace cute;
using B = cutlass::bfloat16_t;
constexpr int D = 128, C = 16, Threads = 128, TmemCols = 256;

template<int N, int K> struct Stage {
  using MMA = decltype(make_tiled_mma(SM100_MMA_F16BF16_SS<
      B,B,float,128,N,UMMA::Major::K,UMMA::Major::K>{}));
  using AS = decltype(partition_shape_A(MMA{}, make_shape(Int<128>{},Int<K>{})));
  using BS = decltype(partition_shape_B(MMA{}, make_shape(Int<N>{},Int<K>{})));
  using AL = decltype(UMMA::tile_to_mma_shape(UMMA::Layout_K_SW32_Atom<B>{},AS{}));
  using BL = decltype(UMMA::tile_to_mma_shape(UMMA::Layout_K_SW32_Atom<B>{},BS{}));
};
constexpr int maximum(int a, int b) { return a > b ? a : b; }
constexpr int ASize = maximum(cosize_v<typename Stage<32,128>::AL>,
                             cosize_v<typename Stage<144,16>::AL>);
constexpr int BSize = maximum(cosize_v<typename Stage<32,128>::BL>,
                             cosize_v<typename Stage<144,16>::BL>);
struct Shared {
  alignas(128) B state[D*D];                 // persistent value-first H[V,K]
  alignas(128) B a[ASize], b[BSize];         // reusable tcgen05 operand layouts
  alignas(128) B rhs[32*128];               // row-major B = actual right operand^T
  alignas(128) B u[D*C];                    // residual, then rounded U^T
  alignas(128) B base_out[D*C];             // rounded Q_d S, value-first
  alignas(128) float acc[D*144];            // visible FP32 results for scalar epilogues
  alignas(16) uint64_t barrier;
  uint32_t tmem;
};

// All 128 threads call. One issuer warp owns the barrier phase; an explicit CTA
// rendezvous prevents phase reuse before every result consumer has completed.
template<int N,int K>
__device__ __forceinline__ void matmul(Shared& s, B const* a, B const* b, int& phase) {
  using T = Stage<N,K>;
  auto mma = typename T::MMA{};
  auto thr = mma.get_slice(Int<0>{});
  auto a_plain = make_tensor(make_smem_ptr(a),make_layout(make_shape(Int<128>{},Int<K>{}),LayoutRight{}));
  auto b_plain = make_tensor(make_smem_ptr(b),make_layout(make_shape(Int<N>{},Int<K>{}),LayoutRight{}));
  auto out = make_tensor(make_smem_ptr(s.acc),make_layout(make_shape(Int<128>{},Int<N>{}),LayoutRight{}));
  auto sa = make_tensor(make_smem_ptr(s.a),typename T::AL{});
  auto sb = make_tensor(make_smem_ptr(s.b),typename T::BL{});
  cooperative_copy<Threads>(threadIdx.x,thr.partition_A(a_plain),sa);
  cooperative_copy<Threads>(threadIdx.x,thr.partition_B(b_plain),sb);
  cutlass::arch::fence_view_async_shared();
  __syncthreads();
  auto ra = thr.make_fragment_A(sa);
  auto rb = thr.make_fragment_B(sb);
  auto out_partition = thr.partition_C(out);
  auto acc = thr.make_fragment_C(out_partition);
  acc.data() = s.tmem;
  if(threadIdx.x < 32) {
    mma.accumulate_ = UMMA::ScaleOut::Zero;
    #pragma unroll
    for(int kk=0; kk<K/16; ++kk) {
      gemm(mma,ra(_,_,kk),rb(_,_,kk),acc);
      mma.accumulate_ = UMMA::ScaleOut::One;
    }
    cutlass::arch::umma_arrive(&s.barrier);
    wait_barrier(s.barrier,phase);
  }
  phase ^= 1;
  __syncthreads();
  auto copy_tmem = make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{},acc);
  auto thread_copy = copy_tmem.get_slice(threadIdx.x);
  auto dst = thread_copy.partition_D(out_partition);
  auto reg = make_tensor<float>(shape(dst));
  copy(copy_tmem,thread_copy.partition_S(acc),reg);
  copy(reg,dst);
  __syncthreads();
}

__device__ __forceinline__ B sigmoid(B x) {
  float h;
  float arg=float(x)*0.5f;
  asm("tanh.approx.f32 %0, %1;" : "=f"(h) : "f"(arg));
  return B(h*0.5f+0.5f);
}

template<bool FuseQK,bool FuseFinal>
__global__ void kernel(B const* v, B const* beta, void const* initial,
    void* final, B* output, B const* kd, B const* qd, B const* kr,
    float const* gt, B const* inv, B const* mqk, int const* prefix,
    int64_t const* cu, int total_t, int heads, int seqs, int total_tiles,
    bool has_initial, bool has_final, bool fp32_state, bool varlen) {
  extern __shared__ __align__(128) char shared_bytes[];
  auto& s=*reinterpret_cast<Shared*>(shared_bytes);
  const int seq=blockIdx.x, head=blockIdx.y;
  const int start=varlen ? int(cu[seq]) : seq*(total_t/seqs);
  const int end=varlen ? int(cu[seq+1]) : start+total_t/seqs;
  const int chunks=(end-start+C-1)/C;
  const int base=varlen ? prefix[seq] : seq*chunks;
  const int64_t state_base=(int64_t(seq)*heads+head)*D*D;
  for(int i=threadIdx.x;i<D*D;i+=Threads) {
    s.state[i]=!has_initial ? B(0.f) : fp32_state ?
      B(static_cast<float const*>(initial)[state_base+i]) :
      static_cast<B const*>(initial)[state_base+i];
  }
  TMEM::Allocator1Sm allocator;
  if(threadIdx.x<32) allocator.allocate(TmemCols,&s.tmem);
  if(threadIdx.x==0) initialize_barrier(s.barrier,1);
  cutlass::arch::fence_barrier_init();
  __syncthreads();
  int phase=0;
  for(int chunk=0;chunk<chunks;++chunk) {
    const int t0=start+chunk*C;
    const int length=min(C,end-t0);
    const int64_t wi=int64_t(head)*total_tiles+base+chunk;
    const B* p_kd=kd+wi*C*D;
    const B* p_qd=qd+wi*C*D;
    const B* p_kr=kr+wi*C*D;
    const B* p_inv=inv+wi*C*C;
    const B* p_mqk=mqk+wi*C*C;
    const float* p_gt=gt+wi*D;

    // H [K_d^T, Q_d^T]. Both products must observe the old persistent state.
    for(int i=threadIdx.x;i<C*D;i+=Threads) {
      s.rhs[i]=p_kd[i];
      if constexpr(FuseQK) s.rhs[C*D+i]=p_qd[i];
    }
    __syncthreads();
    if constexpr(FuseQK) matmul<32,128>(s,s.state,s.rhs,phase);
    else matmul<16,128>(s,s.state,s.rhs,phase);
    for(int i=threadIdx.x;i<D*C;i+=Threads) {
      int value=i/C, token=i%C;
      constexpr int stride=FuseQK ? 32 : 16;
      B projection=B(s.acc[value*stride+token]);
      B vv=token<length ? v[(int64_t(t0+token)*heads+head)*D+value] : B(0.f);
      B bet=token<length ? sigmoid(beta[int64_t(head)*total_t+t0+token]) : B(0.f);
      // Keep BOTH BF16 arithmetic boundaries from the upstream implementation.
      s.u[i]=(vv-projection)*bet;
      if constexpr(FuseQK) s.base_out[i]=B(s.acc[value*32+C+token]);
    }
    __syncthreads();
    if constexpr(!FuseQK) {
      for(int i=threadIdx.x;i<C*D;i+=Threads) s.rhs[i]=p_qd[i];
      __syncthreads();
      matmul<16,128>(s,s.state,s.rhs,phase);
      for(int i=threadIdx.x;i<D*C;i+=Threads) s.base_out[i]=B(s.acc[i]);
      __syncthreads();
    }

    // U^T = residual^T R^T. p_inv already holds R in ordinary row-major order.
    for(int i=threadIdx.x;i<C*C;i+=Threads) s.rhs[i]=p_inv[i];
    __syncthreads();
    matmul<16,16>(s,s.u,s.rhs,phase);
    for(int i=threadIdx.x;i<D*C;i+=Threads) s.u[i]=B(s.acc[i]);
    __syncthreads();

    // U^T [M^T, K_r]. Both depend on rounded U, but not on each other.
    for(int i=threadIdx.x;i<C*C;i+=Threads) s.rhs[i]=p_mqk[i];
    if constexpr(FuseFinal) {
      for(int i=threadIdx.x;i<D*C;i+=Threads) {
        int key=i/C, token=i%C;
        s.rhs[C*C+i]=p_kr[token*D+key];
      }
    }
    __syncthreads();
    if constexpr(FuseFinal) matmul<144,16>(s,s.u,s.rhs,phase);
    else matmul<16,16>(s,s.u,s.rhs,phase);
    for(int i=threadIdx.x;i<D*C;i+=Threads) {
      int value=i/C, token=i%C;
      constexpr int stride=FuseFinal ? 144 : 16;
      B result=s.base_out[i]+B(s.acc[value*stride+token]);
      if(token<length) output[(int64_t(t0+token)*heads+head)*D+value]=result;
    }
    __syncthreads();
    if constexpr(!FuseFinal) {
      for(int i=threadIdx.x;i<D*C;i+=Threads) {
        int key=i/C, token=i%C;
        s.rhs[i]=p_kr[token*D+key];
      }
      __syncthreads();
      matmul<128,16>(s,s.u,s.rhs,phase);
    }
    for(int i=threadIdx.x;i<D*D;i+=Threads) {
      int value=i/D, key=i%D;
      constexpr int stride=FuseFinal ? 144 : 128;
      constexpr int offset=FuseFinal ? C : 0;
      // gate is a vector on the key axis. State persists as BF16 every chunk.
      s.state[i]=B(__fmaf_rn(float(s.state[i]),p_gt[key],s.acc[value*stride+offset+key]));
    }
    __syncthreads();
  }
  if(has_final) for(int i=threadIdx.x;i<D*D;i+=Threads) {
    if(fp32_state) static_cast<float*>(final)[state_base+i]=float(s.state[i]);
    else static_cast<B*>(final)[state_base+i]=s.state[i];
  }
  __syncthreads();
  if(threadIdx.x<32) {
    allocator.release_allocation_lock();
    allocator.free(s.tmem,TmemCols);
  }
}

inline void check(cudaError_t error, char const* action) {
  if(error!=cudaSuccess) throw std::runtime_error(std::string(action)+": "+cudaGetErrorString(error));
}
inline void launch(B const* v,B const* beta,void const* initial,void* final,B* output,
    B const* kd,B const* qd,B const* kr,float const* gt,B const* inv,B const* mqk,
    int const* prefix,int64_t const* cu,int total_t,int heads,int seqs,int total_tiles,
    bool has_initial,bool has_final,bool fp32_state,bool varlen,cudaStream_t stream) {
  auto fn=kernel<C1_FUSE_QK,C1_FUSE_FINAL>;
  check(cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared)),"set shared memory");
  fn<<<dim3(seqs,heads),Threads,sizeof(Shared),stream>>>(v,beta,initial,final,output,
      kd,qd,kr,gt,inv,mqk,prefix,cu,total_t,heads,seqs,total_tiles,
      has_initial,has_final,fp32_state,varlen);
  check(cudaGetLastError(),"C1 nextgen K2 launch");
}
} // namespace c1_nextgen
