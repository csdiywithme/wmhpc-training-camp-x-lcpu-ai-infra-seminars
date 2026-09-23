// Experiment H: BN=128, persistent CTA budget=2/SM, derived from 04g.
// Experiment G: x8 TMEM reads and coalesced FP32 stores via reused shared memory.
// 4.4 combination AB: warp-specialized TMA/MMA plus persistent output-tile loop.
// Derived from 04a_warp_specialized.cu; both source versions are retained.
// 4.4 experiment A: split the 4.3 TMA and MMA control loops between warps.
// Based on 03_pipeline.cu (SHA256 fbec760bebdf20ee005c922d304233aad3e5e0824ac876806c15687588c0b599).
// The host main is copied unchanged except for its printed variant label.
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <random>
#include <vector>
#include "../common.h"

#ifndef STAGES
#define STAGES 3
#endif
constexpr int BM = 128, BN = 128, BK = 64;
constexpr int NSTAGE = STAGES;

__device__ inline uint64_t make_desc_sm100(uint32_t saddr, uint32_t lbo,
                                           uint32_t sbo, uint32_t layout) {
    uint64_t d = 0;
    d |= (uint64_t)((saddr >> 4) & 0x3FFF);
    d |= (uint64_t)((lbo >> 4) & 0x3FFF) << 16;
    d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)layout << 61;
    return d;
}

__device__ inline void mbar_wait(uint32_t mbar, uint32_t phase) {
    uint32_t done = 0;
    while (!done)
        asm volatile(
            "{\n.reg .pred p;\n"
            "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
            "selp.b32 %0, 1, 0, p;\n}"
            : "=r"(done) : "r"(mbar), "r"(phase) : "memory");
}

__device__ inline void issue_tma_tile(uint32_t full, uint32_t dstA,
                                      uint32_t dstB, const CUtensorMap* tmapA,
                                      const CUtensorMap* tmapB, int kBase,
                                      int tileM, int tileN) {
    constexpr uint32_t bytes = 2 * BK * (BM + BN);
    asm volatile(
        "{\n.reg .b64 state;\n"
        "mbarrier.arrive.expect_tx.shared::cta.b64 state, [%0], %1;\n}"
        :: "r"(full), "r"(bytes) : "memory");
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dstA), "l"(tmapA), "r"(kBase), "r"(tileM), "r"(full)
        : "memory");
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dstB), "l"(tmapB), "r"(kBase), "r"(tileN), "r"(full)
        : "memory");
}

__global__ void gemm_pipeline(const __nv_bfloat16* gA, const __nv_bfloat16* gB,
                              float* gD, int M, int N, int K,
                              const __grid_constant__ CUtensorMap tmapA,
                              const __grid_constant__ CUtensorMap tmapB) {
    extern __shared__ uint8_t smem_raw[];
    uint8_t* smem = reinterpret_cast<uint8_t*>(
        (reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
    __shared__ uint32_t tmem_addr_slot;
    __shared__ __align__(8) uint64_t empty[NSTAGE];
    __shared__ __align__(8) uint64_t full[NSTAGE];
    const int tid = threadIdx.x, warp = tid >> 5;
    constexpr uint32_t t_cols = BN;

    if (warp == 0) {
        const uint32_t slot = __cvta_generic_to_shared(&tmem_addr_slot);
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                     :: "r"(slot), "r"(t_cols) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"
                     ::: "memory");
    }
    __syncthreads();
    const uint32_t taddr = tmem_addr_slot;
    const int tile_count = (M / BM) * (N / BN);
    for (int tile_id = blockIdx.x; tile_id < tile_count; tile_id += gridDim.x) {
        if (tid == 0) {
            for (int stage = 0; stage < NSTAGE; ++stage) {
                const uint32_t e = __cvta_generic_to_shared(&empty[stage]);
                const uint32_t f = __cvta_generic_to_shared(&full[stage]);
                asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                             :: "r"(e) : "memory");
                asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                             :: "r"(f) : "memory");
            }
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
        __syncthreads();
        const int tileM = (tile_id / (N / BN)) * BM;
        const int tileN = (tile_id % (N / BN)) * BN;
        const int iters = K / BK;

    // Warp 0 is the producer. It can run ahead by NSTAGE tiles, but a stage
    // cannot be reused until the consumer's asynchronous MMA completes.
    if (tid == 0) {
        for (int it = 0; it < iters; ++it) {
            const int s = it % NSTAGE;
            const uint32_t e = __cvta_generic_to_shared(&empty[s]);
            const uint32_t f = __cvta_generic_to_shared(&full[s]);
            if (it >= NSTAGE)
                mbar_wait(e, ((it - NSTAGE) / NSTAGE) & 1);
            const int stage = s * (BM + BN) * BK * 2;
            const uint32_t a = __cvta_generic_to_shared(smem + stage);
            const uint32_t b = __cvta_generic_to_shared(smem + stage + BM * BK * 2);
            issue_tma_tile(f, a, b, &tmapA, &tmapB, it * BK, tileM, tileN);
        }
    }

    // Warp 1 is the consumer. The other CTA warps wait for the final TMEM
    // readback; tcgen05.mma/commit have single-thread issue granularity.
    if (tid == 32) {
        constexpr uint32_t idesc = (1u << 4) | (1u << 7) | (1u << 10)
                                  | ((BN >> 3) << 17) | ((BM >> 4) << 24);
        for (int it = 0; it < iters; ++it) {
            const int s = it % NSTAGE;
            const uint32_t e = __cvta_generic_to_shared(&empty[s]);
            const uint32_t f = __cvta_generic_to_shared(&full[s]);
            mbar_wait(f, (it / NSTAGE) & 1);
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
            const int stage = s * (BM + BN) * BK * 2;
            const uint32_t a = __cvta_generic_to_shared(smem + stage);
            const uint32_t b = __cvta_generic_to_shared(smem + stage + BM * BK * 2);
            for (int k0 = 0; k0 < BK; k0 += 16) {
                const uint64_t adesc = make_desc_sm100(a + k0 * 2, 0, 1024, 2);
                const uint64_t bdesc = make_desc_sm100(b + k0 * 2, 0, 1024, 2);
                asm volatile(
                    "{ .reg .pred accumulate;\n"
                    "setp.ne.u32 accumulate, %4, 0;\n"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, accumulate;\n}"
                    :: "r"(taddr), "l"(adesc), "l"(bdesc), "r"(idesc),
                       "r"(k0 + it * 4) : "memory");
            }
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
                         :: "r"(e) : "memory");
        }
    }
    __syncthreads();
    // All stage completion barriers must drain before reinitialization.
    for (int stage = 0; stage < NSTAGE; ++stage) {
        const int distance = (iters - 1 - stage + NSTAGE) % NSTAGE;
        const int final_it = iters - 1 - distance;
        if (final_it >= 0)
            mbar_wait(__cvta_generic_to_shared(&empty[stage]),
                      (final_it / NSTAGE) & 1);
    }
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    // All MMA reads of the stage buffers have completed. Reuse that storage
    // as a padded FP32 transpose buffer; no next-tile TMA starts before sync.
    static_assert(BM * (BN + 1) * sizeof(float) <=
                  NSTAGE * (BM + BN) * BK * 2, "epilogue scratch fits stages");
    float* epilogue = reinterpret_cast<float*>(smem);
    for (int n = 0; n < BN; n += 8) {
        const uint32_t read_addr = taddr + ((warp * 32) << 16) + n;
        uint32_t r[8];
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
            "{%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
            : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]),
              "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7])
            : "r"(read_addr) : "memory");
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        #pragma unroll
        for (int j = 0; j < 8; ++j)
            epilogue[tid * (BN + 1) + n + j] = __uint_as_float(r[j]);
    }
    __syncthreads();
    for (int i = tid; i < BM * BN; i += blockDim.x) {
        const int row = i / BN, col = i % BN;
        gD[(tileM + row) * N + tileN + col] = epilogue[row * (BN + 1) + col];
    }
    // Order generic scratch accesses before next tile overwrites it via TMA.
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();
    if (tid == 0) {
        for (int stage = 0; stage < NSTAGE; ++stage) {
            const uint32_t e = __cvta_generic_to_shared(&empty[stage]);
            const uint32_t f = __cvta_generic_to_shared(&full[stage]);
            asm volatile("mbarrier.inval.shared::cta.b64 [%0];"
                         :: "r"(e) : "memory");
            asm volatile("mbarrier.inval.shared::cta.b64 [%0];"
                         :: "r"(f) : "memory");
        }
    }
    __syncthreads();
    }
    if (warp == 0)
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                     :: "r"(taddr), "r"(t_cols) : "memory");
}

int main(int argc, char** argv) {
    int M = argc > 3 ? atoi(argv[1]) : 4096;
    int N = argc > 3 ? atoi(argv[2]) : 4096;
    int K = argc > 3 ? atoi(argv[3]) : 4096;
    if (M % BM || N % BN || K % BK) {
        printf("形状需按 %dx%dx%d 对齐\n", BM, BN, BK);
        return 1;
    }
    size_t nA = (size_t)M * K, nB = (size_t)N * K, nD = (size_t)M * N;
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(-3, 3);
    std::vector<__nv_bfloat16> hA(nA), hB(nB);
    for (auto& v : hA) v = __float2bfloat16((float)dist(rng));
    for (auto& v : hB) v = __float2bfloat16((float)dist(rng));
    __nv_bfloat16 *dA, *dB;
    float *dD, *dRef;
    CUDA_CHECK(cudaMalloc(&dA, nA * 2));
    CUDA_CHECK(cudaMalloc(&dB, nB * 2));
    CUDA_CHECK(cudaMalloc(&dD, nD * 4));
    CUDA_CHECK(cudaMalloc(&dRef, nD * 4));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), nA * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), nB * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dD, 0xFF, nD * 4));

    // TODO:tensor map 从你的 4.2 原样复制。
    alignas(64) CUtensorMap tmapA = {}, tmapB = {};

    // A[M][K]：tensor map 按最内层维度在前描述，即 {K, M}。
    const cuuint64_t globalDimA[2] = {
        static_cast<cuuint64_t>(K), static_cast<cuuint64_t>(M)};
    const cuuint64_t globalStridesA[1] = {
        static_cast<cuuint64_t>(K) * sizeof(__nv_bfloat16)}; // 字节
    const cuuint32_t boxDimA[2] = {BK, BM};                // 元素数
    const cuuint32_t elementStridesA[2] = {1, 1};         // 连续取元素
    // B[N][K]
    const cuuint64_t globalDimB[2] = {
        static_cast<cuuint64_t>(K), static_cast<cuuint64_t>(N)};
    const cuuint64_t globalStridesB[1] = {
        static_cast<cuuint64_t>(K) * sizeof(__nv_bfloat16)}; // 字节
    const cuuint32_t boxDimB[2] = {BK, BN};                // 元素数
    const cuuint32_t elementStridesB[2] = {1, 1};         // 连续取元素

    const CUresult mapAResult = cuTensorMapEncodeTiled(
        &tmapA, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, dA,
        globalDimA, globalStridesA, boxDimA, elementStridesA,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (mapAResult != CUDA_SUCCESS) {
        const char* name = nullptr;
        const char* message = nullptr;
        cuGetErrorName(mapAResult, &name);
        cuGetErrorString(mapAResult, &message);
        fprintf(stderr, "A tensor map encode failed: %s (%d): %s\n",
                name ? name : "unknown", static_cast<int>(mapAResult),
                message ? message : "no error description");
        return 1;
    }

    const CUresult mapBResult = cuTensorMapEncodeTiled(
        &tmapB, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, dB,
        globalDimB, globalStridesB, boxDimB, elementStridesB,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (mapBResult != CUDA_SUCCESS) {
        const char* name = nullptr;
        const char* message = nullptr;
        cuGetErrorName(mapBResult, &name);
        cuGetErrorString(mapBResult, &message);
        fprintf(stderr, "B tensor map encode failed: %s (%d): %s\n",
                name ? name : "unknown", static_cast<int>(mapBResult),
                message ? message : "no error description");
        return 1;
    }

    cudaDeviceProp device = {};
    CUDA_CHECK(cudaGetDeviceProperties(&device, 0));
    const int total_tiles = (M / BM) * (N / BN);
    const int resident_grid = device.multiProcessorCount * 2; // S=3 shared limit for BN=128
    dim3 grid(total_tiles < resident_grid ? total_tiles : resident_grid);
    // Dynamic shared memory includes all pipeline stages and alignment slack.
    size_t smemBytes = (size_t)NSTAGE * (BM + BN) * BK * 2 + 1024;
    CUDA_CHECK(cudaFuncSetAttribute(gemm_pipeline,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)smemBytes));
    auto launch = [&] {
        gemm_pipeline<<<grid, 128, smemBytes>>>(dA, dB, dD, M, N, K, tmapA,
                                                tmapB);
    };
    launch();
    CUDA_CHECK_KERNEL();

    cublasHandle_t h;
    cublasCreate(&h);
    float alpha = 1.f, beta = 0.f;
    cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, dB, CUDA_R_16BF,
                 K, dA, CUDA_R_16BF, K, &beta, dRef, CUDA_R_32F, N,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> got(nD), ref(nD);
    CUDA_CHECK(cudaMemcpy(got.data(), dD, nD * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ref.data(), dRef, nD * 4, cudaMemcpyDeviceToHost));
    long bad = 0;
    for (size_t i = 0; i < nD; i++) bad += got[i] != ref[i];

    int iters = (size_t)M * N >= (size_t)4096 * 4096 ? 20 : 100;
    float ms = time_avg_ms(launch, iters);
    double tflops = 2.0 * M * N * K / (ms * 1e9);
    float cub_ms = time_avg_ms(
        [&] {
            cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, dB,
                         CUDA_R_16BF, K, dA, CUDA_R_16BF, K, &beta, dRef,
                         CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT);
        },
        iters);
    double cub_tflops = 2.0 * M * N * K / (cub_ms * 1e9);
    printf("[4.4 epilogue BN=128 S=%d] M=%d N=%d K=%d  %s(bad=%ld)  %.2f ms  %.1f "
           "TFLOPS  (cuBLAS %.1f, 达成率 %.0f%%)\n",
           NSTAGE, M, N, K, bad ? "FAIL" : "PASS", bad, ms, tflops,
           cub_tflops, 100.0 * tflops / cub_tflops);
    cublasDestroy(h);
    return bad != 0;
}
