// C1 isolated shared-resident BF16 tile experiment. Inspired by the interface
// documented in NVIDIA CUTLASS examples/cute/tutorial/blackwell/01_mma_sm100.cu.
// Each timed iteration accumulates A*B^T from shared memory into the previous
// FP32 accumulator; output is read back only after the last iteration. The true
// data dependency prevents the compiler hoisting repeated identical GEMMs.
// This does not measure a complete KDA pipeline.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cute/tensor.hpp>
#include <cute/algorithm/cooperative_copy.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cutlass/arch/barrier.h>
using namespace cute;
using BF16 = cutlass::bfloat16_t;
#define CHECK(x) do { cudaError_t c1_cuda_status=(x); if(c1_cuda_status!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(c1_cuda_status));exit(2);} }while(0)

template<int M,int N,int K>
struct TcTypes {
  using MMA = decltype(make_tiled_mma(SM100_MMA_F16BF16_SS<BF16,BF16,float,M,N,UMMA::Major::K,UMMA::Major::K>{}));
  using AS = decltype(partition_shape_A(MMA{},make_shape(Int<M>{},Int<K>{})));
  using BS = decltype(partition_shape_B(MMA{},make_shape(Int<N>{},Int<K>{})));
  using AL = decltype(UMMA::tile_to_mma_shape(UMMA::Layout_K_SW32_Atom<BF16>{},AS{}));
  using BL = decltype(UMMA::tile_to_mma_shape(UMMA::Layout_K_SW32_Atom<BF16>{},BS{}));
  struct Shared {
    alignas(128) BF16 a[cosize_v<AL>], b[cosize_v<BL>];
    alignas(16) uint64_t barrier;
    uint32_t tmem;
  };
};

template<int M,int N,int K>
__global__ void tcgen_tile(BF16 const* a,BF16 const* b,float* out,int loops) {
  using T=TcTypes<M,N,K>;
  extern __shared__ __align__(128) char sm[];
  auto& s=*reinterpret_cast<typename T::Shared*>(sm);
  auto mma=typename T::MMA{};
  auto thr=mma.get_slice(Int<0>{});
  auto ga=make_tensor(make_gmem_ptr(a),make_layout(make_shape(Int<M>{},Int<K>{}),LayoutRight{}));
  auto gb=make_tensor(make_gmem_ptr(b),make_layout(make_shape(Int<N>{},Int<K>{}),LayoutRight{}));
  auto gc=make_tensor(make_gmem_ptr(out+blockIdx.x*M*N),make_layout(make_shape(Int<M>{},Int<N>{}),LayoutRight{}));
  auto sa=make_tensor(make_smem_ptr(s.a),typename T::AL{});
  auto sb=make_tensor(make_smem_ptr(s.b),typename T::BL{});
  cooperative_copy<128>(threadIdx.x,thr.partition_A(ga),sa);
  cooperative_copy<128>(threadIdx.x,thr.partition_B(gb),sb);
  TMEM::Allocator1Sm alloc;
  constexpr int COLS = N < 32 ? 32 : N;
  if(threadIdx.x<32) alloc.allocate(COLS,&s.tmem);
  if(threadIdx.x==0) initialize_barrier(s.barrier,1);
  cutlass::arch::fence_barrier_init();
  cutlass::arch::fence_view_async_shared();
  __syncthreads();
  auto ra=thr.make_fragment_A(sa);
  auto rb=thr.make_fragment_B(sb);
  auto cg=thr.partition_C(gc);
  auto acc=thr.make_fragment_C(cg);
  acc.data()=s.tmem;
  int phase=0;
  mma.accumulate_=UMMA::ScaleOut::Zero;
  // Only the issuer warp reuses this barrier's phases. Other warps rendezvous
  // after all repeats, so a delayed consumer cannot miss multiple phases.
  if(threadIdx.x<32) {
  for(int repeat=0;repeat<loops;repeat++) {
      #pragma unroll
      for(int kk=0;kk<K/16;kk++) {
        gemm(mma,ra(_,_,kk),rb(_,_,kk),acc);
        mma.accumulate_=UMMA::ScaleOut::One;
      }
      cutlass::arch::umma_arrive(&s.barrier);
    wait_barrier(s.barrier,phase);
    phase^=1;
  }
  }
  __syncthreads();
  using TmemLoad = std::conditional_t<M == 64, SM100_TMEM_LOAD_16dp64b1x, SM100_TMEM_LOAD_32dp32b1x>;
  auto cp=make_tmem_copy(TmemLoad{},acc);
  auto ct=cp.get_slice(threadIdx.x);
  auto dst=ct.partition_D(cg);
  auto reg=make_tensor<float>(shape(dst));
  copy(cp,ct.partition_S(acc),reg);
  copy(reg,dst);
  __syncthreads();
  if(threadIdx.x<32) {alloc.release_allocation_lock();alloc.free(s.tmem,COLS);}
}

template<int M,int N,int K>
__global__ void sm80_tile(BF16 const* a,BF16 const* b,float* out,int loops) {
  using AL=decltype(tile_to_shape(GMMA::Layout_K_INTER_Atom<BF16>{},make_shape(Int<M>{},Int<K>{}),LayoutLeft{}));
  using BL=decltype(tile_to_shape(GMMA::Layout_K_INTER_Atom<BF16>{},make_shape(Int<N>{},Int<K>{}),LayoutLeft{}));
  __shared__ __align__(128) BF16 aa[cosize_v<AL>],bb[cosize_v<BL>];
  auto sa=make_tensor(make_smem_ptr(aa),AL{});
  auto sb=make_tensor(make_smem_ptr(bb),BL{});
  for(int i=threadIdx.x;i<M*K;i+=128)sa(i/K,i%K)=a[i];
  for(int i=threadIdx.x;i<N*K;i+=128)sb(i/K,i%K)=b[i];
  __syncthreads();
  auto mma=make_tiled_mma(MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>{},Layout<Shape<_1,_1>>{},Tile<_16,_16,_16>{});
  auto thr=mma.get_slice(threadIdx.x%32);
  auto ca=make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N,BF16>{},mma);
  auto cb=make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N,BF16>{},mma);
  auto ta=ca.get_slice(threadIdx.x%32);
  auto tb=cb.get_slice(threadIdx.x%32);
  auto gc=make_tensor(make_gmem_ptr(out+blockIdx.x*M*N),make_layout(make_shape(Int<M>{},Int<N>{}),LayoutRight{}));
  // Assign independent 16x16 output blocks cyclically to four warps.
  for(int tile=threadIdx.x/32;tile<(M/16)*(N/16);tile+=4) {
    int mi=tile/(N/16),ni=tile%(N/16);
    auto ar=local_tile(sa,Shape<_16,_16>{},make_coord(mi,0));
    auto br=local_tile(sb,Shape<_16,_16>{},make_coord(ni,0));
    auto cr=local_tile(gc,Shape<_16,_16>{},make_coord(mi,ni));
    auto ra=thr.partition_fragment_A(ar);
    auto rb=thr.partition_fragment_B(br);
    auto acc=thr.make_fragment_C(thr.partition_C(cr));
    clear(acc);
    for(int repeat=0;repeat<loops;repeat++) {
      #pragma unroll
      for(int kk=0;kk<K/16;kk++) {
        auto ab=local_tile(sa,Shape<_16,_16>{},make_coord(mi,kk));
        auto bbv=local_tile(sb,Shape<_16,_16>{},make_coord(ni,kk));
        copy(ca,ta.partition_S(ab),ta.retile_D(ra));
        copy(cb,tb.partition_S(bbv),tb.retile_D(rb));
        gemm(thr,ra,rb,acc);
      }
    }
    copy(acc,thr.partition_C(cr));
  }
}

template<int M,int N,int K>
void run_case(int blocks,int loops) {
  std::vector<BF16> a(M*K),b(N*K);
  for(int i=0;i<M*K;i++)a[i]=BF16(float((i*13)%17-8)/16);
  for(int i=0;i<N*K;i++)b[i]=BF16(float((i*7)%19-9)/16);
  BF16 *da,*db;float *out;
  CHECK(cudaMalloc(&da,a.size()*2));CHECK(cudaMalloc(&db,b.size()*2));CHECK(cudaMalloc(&out,blocks*M*N*4));
  CHECK(cudaMemcpy(da,a.data(),a.size()*2,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(db,b.data(),b.size()*2,cudaMemcpyHostToDevice));
  constexpr int smem=sizeof(typename TcTypes<M,N,K>::Shared);
  CHECK(cudaFuncSetAttribute(tcgen_tile<M,N,K>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
  cudaEvent_t s,e;CHECK(cudaEventCreate(&s));CHECK(cudaEventCreate(&e));
  for(int kind=0;kind<2;kind++) {
    auto launch=[&](){if(kind==0)sm80_tile<M,N,K><<<blocks,128>>>(da,db,out,loops);else tcgen_tile<M,N,K><<<blocks,128,smem>>>(da,db,out,loops);};
    CHECK(cudaMemset(out,0xff,blocks*M*N*4)); // NaN sentinel catches unwritten elements.
    launch();CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
    std::vector<float> actual(blocks*M*N);CHECK(cudaMemcpy(actual.data(),out,blocks*M*N*4,cudaMemcpyDeviceToHost));
    std::vector<float> gold(M*N);
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){float v=0;for(int k=0;k<K;k++)v+=float(a[m*K+k])*float(b[n*K+k]);gold[m*N+n]=v*loops;}
    float error=0;
    for(int i=0;i<blocks*M*N;i++){
      if(!std::isfinite(actual[i])){fprintf(stderr,"NONFINITE kind=%d element=%d\n",kind,i);exit(4);}
      error=fmaxf(error,fabsf(gold[i%(M*N)]-actual[i]));
    }
    if(error>1e-4f){fprintf(stderr,"FAIL kind=%d M=%d N=%d K=%d max=%g\n",kind,M,N,K,error);exit(3);}
    cudaStream_t stream;CHECK(cudaStreamCreate(&stream));
    cudaGraph_t graph;cudaGraphExec_t exec;
    CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
    for(int j=0;j<20;j++){
      if(kind==0)sm80_tile<M,N,K><<<blocks,128,0,stream>>>(da,db,out,loops);
      else tcgen_tile<M,N,K><<<blocks,128,smem,stream>>>(da,db,out,loops);
    }
    CHECK(cudaStreamEndCapture(stream,&graph));CHECK(cudaGraphInstantiate(&exec,graph,0));
    for(int j=0;j<5;j++)CHECK(cudaGraphLaunch(exec,0));CHECK(cudaDeviceSynchronize());
    for(int r=0;r<5;r++) {
      CHECK(cudaEventRecord(s));for(int j=0;j<20;j++)CHECK(cudaGraphLaunch(exec,0));CHECK(cudaEventRecord(e));CHECK(cudaEventSynchronize(e));
      float ms;CHECK(cudaEventElapsedTime(&ms,s,e));
      printf("{\"kind\":\"%s\",\"M\":%d,\"N\":%d,\"K\":%d,\"blocks\":%d,\"inner_loops\":%d,\"repeat\":%d,\"timing\":\"cuda_graph_20x20\",\"kernel_us\":%.6f,\"max_error\":%.6f}\n",kind?"tcgen05":"sm80",M,N,K,blocks,loops,r,ms*1000/400,error);
    }
    CHECK(cudaGraphExecDestroy(exec));CHECK(cudaGraphDestroy(graph));CHECK(cudaStreamDestroy(stream));
  }
  CHECK(cudaEventDestroy(s));CHECK(cudaEventDestroy(e));CHECK(cudaFree(da));CHECK(cudaFree(db));CHECK(cudaFree(out));
}
int main(){
  for(int blocks:{1,12,96,148})for(int loops:{1,64}){
    run_case<128,16,128>(blocks,loops); // transposed C16 x D128 state products
    run_case<128,16,16>(blocks,loops);  // transposed INV@R / Mqk@U
    run_case<128,128,16>(blocks,loops);// state update
    run_case<64,16,16>(blocks,loops);  // padded small tile: 25% useful M for C1 inverse
  }
}
