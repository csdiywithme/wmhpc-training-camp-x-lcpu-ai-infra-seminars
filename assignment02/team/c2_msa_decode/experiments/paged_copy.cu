// C2 discussion 3: two SM-side index loads followed by a regular TMA page copy.
// Full allocated pages only. This does not implement attention or causal masks.
// Compare identical global -> shared -> global work; graph replay uses warm data.
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>

#define CK(x) do { auto e=(x); if(e!=cudaSuccess) { fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e)); exit(1); } } while(0)
#define DR(x) do { auto e=(x); if(e!=CUDA_SUCCESS) { const char* s; cuGetErrorString(e,&s); fprintf(stderr,"%s: %s\n",#x,s); exit(1); } } while(0)
constexpr int PAGE=128, WIDTH=256, TOPK=16, BLOCKS=64;

__device__ inline void wait_bar(uint32_t b) {
    uint32_t done=0;
    while(!done) asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p,[%1],0; selp.b32 %0,1,0,p; }" : "=r"(done):"r"(b):"memory");
}

template<int BYTES, bool TMA>
__global__ void paged_copy(const uint8_t* src, uint8_t* dst, const int* topk,
                          const int* table, int heads,
                          const __grid_constant__ CUtensorMap map) {
    extern __shared__ __align__(128) uint8_t sm[];
    __shared__ __align__(8) uint64_t barrier;
    __shared__ int phys;
    constexpr int nbytes=PAGE*WIDTH*BYTES;
    int slot=blockIdx.x%TOPK;
    int req=blockIdx.x/TOPK;
    int head=blockIdx.y;
    if(threadIdx.x==0) {
        // The tensor map does NOT carry out this pointer chase.
        int logical=topk[(req*heads+head)*TOPK+slot];
        phys=table[req*BLOCKS+logical];
        if constexpr(TMA) {
            uint32_t bar=uint32_t(__cvta_generic_to_shared(&barrier));
            asm volatile("mbarrier.init.shared::cta.b64 [%0],1;"::"r"(bar):"memory");
            asm volatile("fence.proxy.async.shared::cta;":::"memory");
        }
    }
    __syncthreads();
    if constexpr(TMA) {
        uint32_t bar=uint32_t(__cvta_generic_to_shared(&barrier));
        if(threadIdx.x==0) {
            uint32_t buf=uint32_t(__cvta_generic_to_shared(sm));
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(bar),"r"(nbytes):"memory");
            asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1,{0,0,%2,%3}],[%4];"
                         ::"r"(buf),"l"(&map),"r"(head),"r"(phys),"r"(bar):"memory");
        }
        wait_bar(bar);
    } else {
        const uint4* in=reinterpret_cast<const uint4*>(src+(size_t(phys)*heads+head)*nbytes);
        uint4* buf=reinterpret_cast<uint4*>(sm);
        for(int i=threadIdx.x;i<nbytes/16;i+=blockDim.x) buf[i]=in[i];
        __syncthreads();
    }
    // Identical observable output also prevents removal of either copy path.
    uint4* out=reinterpret_cast<uint4*>(dst+((size_t(req)*heads+head)*TOPK+slot)*nbytes);
    const uint4* buf=reinterpret_cast<const uint4*>(sm);
    for(int i=threadIdx.x;i<nbytes/16;i+=blockDim.x) out[i]=buf[i];
    if constexpr(TMA) {
        __syncthreads();
        if(threadIdx.x==0) {
            uint32_t bar=uint32_t(__cvta_generic_to_shared(&barrier));
            asm volatile("mbarrier.inval.shared::cta.b64 [%0];"::"r"(bar):"memory");
        }
    }
}

template<class F> std::vector<float> graph_times(F launch) {
    for(int i=0;i<5;++i) launch();
    CK(cudaDeviceSynchronize());
    cudaStream_t stream; CK(cudaStreamCreate(&stream));
    cudaGraph_t graph; cudaGraphExec_t exec;
    CK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
    // launch accepts a stream so all repeated nodes are captured on that stream.
    for(int i=0;i<100;++i) launch(stream);
    CK(cudaStreamEndCapture(stream,&graph));
    CK(cudaGraphInstantiate(&exec,graph,0));
    cudaEvent_t a,b; CK(cudaEventCreate(&a));CK(cudaEventCreate(&b));
    std::vector<float> us;
    for(int r=0;r<9;++r) {
        CK(cudaEventRecord(a,stream)); CK(cudaGraphLaunch(exec,stream)); CK(cudaEventRecord(b,stream));
        CK(cudaEventSynchronize(b));float ms;CK(cudaEventElapsedTime(&ms,a,b));us.push_back(ms*10);
    }
    CK(cudaEventDestroy(a));CK(cudaEventDestroy(b));CK(cudaGraphExecDestroy(exec));
    CK(cudaGraphDestroy(graph));CK(cudaStreamDestroy(stream));return us;
}

template<int BYTES> bool run(int batch,int heads) {
    constexpr int nbytes=PAGE*WIDTH*BYTES;
    int pages=batch*BLOCKS;
    size_t nsrc=size_t(pages)*heads*nbytes, nout=size_t(batch)*heads*TOPK*nbytes;
    std::mt19937 rng(20260913+batch+heads);
    std::vector<uint8_t> src(nsrc), out(nout), expected(nout);
    for(size_t i=0;i<nsrc;++i) src[i]=uint8_t(rng());
    std::vector<int> table(pages),topk(batch*heads*TOPK);
    std::iota(table.begin(),table.end(),0);std::shuffle(table.begin(),table.end(),rng);
    for(int r=0;r<batch;++r) for(int h=0;h<heads;++h) {
        std::vector<int> slots(BLOCKS);std::iota(slots.begin(),slots.end(),0);std::shuffle(slots.begin(),slots.end(),rng);
        for(int k=0;k<TOPK;++k) {
            int logical=slots[k], physical=table[r*BLOCKS+logical];
            topk[(r*heads+h)*TOPK+k]=logical;
            std::copy_n(src.data()+(size_t(physical)*heads+h)*nbytes,nbytes,
                        expected.data()+((size_t(r)*heads+h)*TOPK+k)*nbytes);
        }
    }
    uint8_t *di,*dout;int *dt,*db;
    CK(cudaMalloc(&di,nsrc));CK(cudaMalloc(&dout,nout));
    CK(cudaMalloc(&dt,topk.size()*4));CK(cudaMalloc(&db,table.size()*4));
    CK(cudaMemcpy(di,src.data(),nsrc,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dt,topk.data(),topk.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(db,table.data(),table.size()*4,cudaMemcpyHostToDevice));
    CUtensorMap map={};uint64_t dims[]={WIDTH,PAGE,uint64_t(heads),uint64_t(pages)};
    uint64_t strides[]={WIDTH*BYTES,nbytes,uint64_t(nbytes)*heads};
    uint32_t box[]={WIDTH,PAGE,1,1}, element[]={1,1,1,1};
    DR(cuTensorMapEncodeTiled(&map,BYTES==1?CU_TENSOR_MAP_DATA_TYPE_UINT8:CU_TENSOR_MAP_DATA_TYPE_UINT16,
        4,di,dims,strides,box,element,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CK(cudaFuncSetAttribute(paged_copy<BYTES,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,nbytes));
    CK(cudaFuncSetAttribute(paged_copy<BYTES,false>,cudaFuncAttributeMaxDynamicSharedMemorySize,nbytes));
    dim3 grid(batch*TOPK,heads);
    auto sm=[&](cudaStream_t stream=0){paged_copy<BYTES,false><<<grid,128,nbytes,stream>>>(di,dout,dt,db,heads,map);};
    auto tma=[&](cudaStream_t stream=0){paged_copy<BYTES,true><<<grid,128,nbytes,stream>>>(di,dout,dt,db,heads,map);};
    bool good=true;
    for(bool use_tma:{false,true}) {
        CK(cudaMemset(dout,0xA5,nout));if(use_tma)tma();else sm();
        CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(out.data(),dout,nout,cudaMemcpyDeviceToHost));
        size_t bad=0;for(size_t i=0;i<nout;++i)bad+=out[i]!=expected[i];good&=bad==0;
        auto times=use_tma?graph_times(tma):graph_times(sm);auto sorted=times;std::sort(sorted.begin(),sorted.end());
        printf("{\"batch\":%d,\"heads\":%d,\"element_bytes\":%d,\"path\":\"%s\",\"mismatches\":%zu,\"copy_bytes_read_plus_write\":%zu,\"median_us\":%.6f,\"samples_us\":[",batch,heads,BYTES,use_tma?"tma":"sm_vector",bad,2*nout,sorted[4]);
        for(size_t i=0;i<times.size();++i)printf("%s%.6f",i?",":"",times[i]);puts("]}");fflush(stdout);
    }
    CK(cudaFree(di));CK(cudaFree(dout));CK(cudaFree(dt));CK(cudaFree(db));return good;
}

int main() {
    cudaDeviceProp p;CK(cudaGetDeviceProperties(&p,0));printf("{\"gpu\":\"%s\",\"cc\":\"%d.%d\",\"sms\":%d,\"scope\":\"warm graph full-page copy, not attention\"}\n",p.name,p.major,p.minor,p.multiProcessorCount);
    bool good=true;for(int h:{1,4})for(int b:{1,4,8,16}){good&=run<1>(b,h);good&=run<2>(b,h);}return good?0:1;
}
