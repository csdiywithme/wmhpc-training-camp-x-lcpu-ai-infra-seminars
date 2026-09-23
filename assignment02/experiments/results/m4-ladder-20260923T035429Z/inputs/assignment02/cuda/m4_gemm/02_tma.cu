// 问题 4.2(MODIFY):把 4.1 的 staging 换成 TMA,其余不动(仍单缓冲)。
//
// 从你自己的 01_tiled.cu 出发:mma 发射、epilogue、判测口径全部不变,
// 改动集中在两处——host 侧建 tensor map,kernel 侧把 st.shared staging
// 换成 cp.async.bulk.tensor + mbarrier。
//
// 直接告知的事实(工具链与布局配对,不属于考核点):
//   - tensor map 用驱动 API cuTensorMapEncodeTiled 建(Makefile 已链
//     -lcuda);kernel 参数按 const __grid_constant__ CUtensorMap 传
//   - 维度次序:dim0 是最内维(这里是 K,单位为元素数);globalStrides
//     只填外维的字节跨度 {K*2};box 是一次搬运的块 {BK, BM}(B 矩阵
//     {BK, BN});elementStrides 全 1
//   - swizzle 选 CU_TENSOR_MAP_SWIZZLE_128B:TMA 硬件落进 smem 的布局
//     与你 4.1 手工 swz128 摆出来的完全相同,descriptor 一个字段都
//     不用改;interleave/L2 promotion/oob fill 都取 NONE
//   - fence 口径(2.1(b) 在这里兑现):TMA 写 smem 与 tcgen05 读 smem
//     都走 async proxy,fence.proxy.async 不再需要;mbar_wait 之后的
//     tcgen05.fence::after_thread_sync 仍然要
//
// 交付:PASS + 梯子表第二行;回答 handout 4.2 的问题(相对 4.1 的提升
// 为什么这么大——4.1 的 staging 成本由什么构成,用 ncu 佐证)。
//
// 运行:make run/m4_gemm/02_tma;自定形状 ./bin/m4_gemm/02_tma M N K
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <random>
#include <vector>
#include "../common.h"

constexpr int BM = 128, BN = 64, BK = 64;

// SM100 shared memory descriptor；swizzle 由 TMA 完成。
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
            : "=r"(done)
            : "r"(mbar), "r"(phase));
}

__global__ void gemm_tma(const __nv_bfloat16* gA, const __nv_bfloat16* gB,
                         float* gD, int M, int N, int K,
                         const __grid_constant__ CUtensorMap tmapA,
                         const __grid_constant__ CUtensorMap tmapB) {
    extern __shared__ uint8_t smem_raw[];
    uint8_t* smem =
        (uint8_t*)(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023);

    // 初始化 full（TMA 完成）、empty（MMA 完成）及 TMEM。
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid % 32;

    __shared__ uint32_t tmem_addr_slot;
    __shared__ __align__(8) uint64_t barrier;
    __shared__ __align__(8) uint64_t barrier_full;
    const uint32_t slot_saddr = static_cast<uint32_t>(__cvta_generic_to_shared(&tmem_addr_slot));
    const uint32_t mbar = static_cast<uint32_t>(__cvta_generic_to_shared(&barrier));
    const uint32_t mbar_full = static_cast<uint32_t>(__cvta_generic_to_shared(&barrier_full));
    constexpr uint32_t t_cols = BN; // M = 128, FP32

    if (tid == 0){
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                    :: "r"(mbar) : "memory");
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                    :: "r"(mbar_full) : "memory");

    }
    if (warp == 0) {
        // 输入是 shared 地址；返回的 TMEM 地址被写到 shared 变量里。
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                     :: "r"(slot_saddr), "r"(t_cols) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"
                     ::: "memory"); // 后面不再分配；这不是释放内存。
    }
    __syncthreads();
    const uint32_t taddr = tmem_addr_slot;

    // 本 block 的输出 tile。
    const int tileM = blockIdx.x*BM;
    const int tileN = blockIdx.y*BN;
    const int sa_ofst = 0;
    const int sb_ofst = BM*BK*2;
    // A/B 在 shared memory 中的偏移单位为字节。
    __nv_bfloat16* sA = reinterpret_cast<__nv_bfloat16*>(smem + sa_ofst);
    __nv_bfloat16* sB = reinterpret_cast<__nv_bfloat16*>(smem + sb_ofst);
    for (int it = 0; it < K/BK; it += 1){
        if (tid == 0) {
            const uint32_t bytes = 2 * BK * (BM + BN);

            asm volatile(
                "{\n"
                "  .reg .b64 state;\n"
                "  mbarrier.arrive.expect_tx.shared::cta.b64 "
                "state, [%0], %1;\n"
                "}\n"
                :
                : "r"(mbar_full), "r"(bytes)
                : "memory"
            );
            const uint32_t dstA = static_cast<uint32_t>(__cvta_generic_to_shared(sA));
            const uint32_t dstB = static_cast<uint32_t>(__cvta_generic_to_shared(sB));
            const int kBase = it * BK;

            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cluster.global"
                ".mbarrier::complete_tx::bytes "
                "[%0], [%1, {%2, %3}], [%4];"
                :
                : "r"(dstA),
                "l"(&tmapA),
                "r"(kBase),
                "r"(tileM),
                "r"(mbar_full)
                : "memory"
            );
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cluster.global"
                ".mbarrier::complete_tx::bytes "
                "[%0], [%1, {%2, %3}], [%4];"
                :
                : "r"(dstB),
                "l"(&tmapB),
                "r"(kBase),
                "r"(tileN),
                "r"(mbar_full)
                : "memory"
            );
        }

        // 等 A/B 搬运完成，再交给 MMA。
        mbar_wait(mbar_full, it & 1);
        __syncthreads();
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        // 每轮发射 4 条 k16 MMA。
        if (tid == 0) {
            const uint32_t a_base = static_cast<uint32_t>(__cvta_generic_to_shared(sA));
            const uint32_t b_base = static_cast<uint32_t>(__cvta_generic_to_shared(sB));
            // idesc: D=F32(1), A/B=BF16(1), M/16, N/8。
            // transpose=0 表示 K-major；dense、negate 等位均为 0。
            constexpr uint32_t idesc = (1u << 4) | (1u << 7) | (1u << 10)
                                    | ((BN >> 3) << 17) | ((BM >> 4) << 24);
            for (int k0 = 0; k0 < BK; k0 += 16) {
                // 8 行 * 每行 128B = SBO 1024B；K-major swizzle 不使用 LBO。
                const uint64_t adesc = make_desc_sm100(a_base + k0 * 2, 0, 1024, 2);
                const uint64_t bdesc = make_desc_sm100(b_base + k0 * 2, 0, 1024, 2);
                // 整个 K 循环仅第一条 MMA 禁用旧 D，其余累加。
                asm volatile(
                    "{ .reg .pred accumulate;\n"
                    "  setp.ne.u32 accumulate, %4, 0;\n"
                    "  tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, accumulate;\n"
                    "}"
                    :: "r"(taddr), "l"(adesc), "l"(bdesc), "r"(idesc), "r"(k0 + it * 4)
                    : "memory");
            }
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
                    :: "r"(mbar) : "memory");
        }
        // 等 MMA 消费完成，下一轮才可覆盖 shared memory。
        mbar_wait(mbar, it & 1);
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    }
    // 从 TMEM 读回并写入 D，global 行跨度为 N。
    for (int n = 0; n < BN; ++n) {
        const uint32_t read_addr = taddr + ((warp * 32) << 16) + n;
        uint32_t bits;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];"
                     : "=r"(bits) : "r"(read_addr) : "memory");
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        gD[tileM*N + tileN + (warp * 32 + lane) * N + n] = __uint_as_float(bits);
    }

    // 释放 TMEM。
    __syncthreads();
    if (warp == 0)
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                     :: "r"(taddr), "r"(t_cols) : "memory");
    if (tid == 0)
        asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"(mbar) : "memory");

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

    // 创建 A/B tensor map。
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

    dim3 grid(M / BM, N / BN);
    size_t smemBytes = (size_t)(BM + BN) * BK * 2 + 1024;
    CUDA_CHECK(cudaFuncSetAttribute(gemm_tma,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)smemBytes));
    auto launch = [&] {
        gemm_tma<<<grid, 128, smemBytes>>>(dA, dB, dD, M, N, K, tmapA, tmapB);
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
    printf("[4.2 tma] M=%d N=%d K=%d  %s(bad=%ld)  %.2f ms  %.1f TFLOPS  "
           "(cuBLAS %.1f, 达成率 %.0f%%)\n",
           M, N, K, bad ? "FAIL" : "PASS", bad, ms, tflops, cub_tflops,
           100.0 * tflops / cub_tflops);
    cublasDestroy(h);
    return bad != 0;
}
