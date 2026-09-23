// 问题 3.2(模块压轴):从零写 tcgen05 单 tile GEMM。
//
// 形状 m128n64k64,bf16 输入,f32 累加,cta_group::1,单 block 128 线程。
// 数据通路:global -> smem(K-major + 128B swizzle)-> tcgen05.mma ->
// TMEM -> tcgen05.ld -> global。判测(main 已给出)用小整数严格对拍。
//
// 给你的材料:课件 F27 的七步流程(下面 kernel 里只留了步骤注释)、
// 你在 2.2 写的 descriptor 编码(SM100 位域)、2.3 的 swizzle_128B
// (staging 布局用它;布局错,结果必错——这里是它的真硬件判测)。
// 其余(TMEM alloc、mbarrier、idesc、tcgen05.mma/ld 的写法)自己查
// PTX ISA 对应章节,课件 C15-C21 讲过每一件的语义,数字换成本题形状。
//
// 两个提醒,直接说明:
// - smem 写完到发射 mma 之间需要 fence.proxy.async(2.1 排序题的答案
//   在这里上真硬件;漏掉的现象自己观察一次,写进报告)
// - tcgen05.ld 每个 warp 只能读自己的 32 条 lane(3.1(a));taddr 高
//   16 bit 是 lane 偏移、低 16 bit 是列偏移;ld 之后要 tcgen05.wait::ld
//
// 运行:make run/m3_tcgen05/02_single_tile;多 seed:./judge_tile.sh
#include <cuda_bf16.h>
#include <cstdio>
#include <random>
#include <cstdint>
#include <vector>
#include "../common.h"

constexpr int M = 128, N = 64, K = 64;

// 128B swizzle 的物理偏移(即 2.3 的 swizzle_128B;row 是 K-major 下的
// 行 = M 或 N 维,col 是 K 维字节)。atom = 8 行 × 128B,SBO=1024。
__host__ __device__ inline int swz128(int row, int colByte) {
    int atom = row >> 3, r = row & 7, chunk = colByte >> 4, in16 = colByte & 15;
    return atom * 1024 + r * 128 + ((chunk ^ r) << 4) + in16;
}

__device__ inline uint64_t make_desc_sm100(uint32_t saddr, uint32_t lbo,
                                           uint32_t sbo, uint32_t layout) {
    uint64_t d = 0;
    d |= (uint64_t)((saddr >> 4) & 0x3FFF);
    d |= (uint64_t)((lbo >> 4) & 0x3FFF) << 16;
    d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;
    d |= (uint64_t)1 << 46;             // version = 1(SM100)
    d |= (uint64_t)layout << 61;        // 3 bit layout type
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
            : "r"(mbar), "r"(phase) : "memory");
}

__global__ void tcgen05_tile(const __nv_bfloat16* gA, const __nv_bfloat16* gB,
                             float* gD) {
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    // 1024B 对齐让 128B swizzle 的 base offset 为 0。
    __shared__ __align__(1024) __nv_bfloat16 sA[M * K];
    __shared__ __align__(1024) __nv_bfloat16 sB[N * K];
    __shared__ uint32_t tmem_addr_slot;
    __shared__ __align__(8) uint64_t barrier;
    const uint32_t slot_saddr = static_cast<uint32_t>(__cvta_generic_to_shared(&tmem_addr_slot));
    const uint32_t mbar = static_cast<uint32_t>(__cvta_generic_to_shared(&barrier));
    constexpr uint32_t t_cols = N; // M=128, FP32: 一列放 128 个输出，需 N 列。

    // (1) init 是单线程操作；alloc 是整个 warp 的集体操作。
    // count=1：后面只有一次 commit 的 arrive::one 通知，不是 128 个等待者。
    if (tid == 0)
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                     :: "r"(mbar) : "memory");
    if (warp == 0) {
        // 输入是 shared 地址；返回的 TMEM 地址被写到 shared 变量里。
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                     :: "r"(slot_saddr), "r"(t_cols) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"
                     ::: "memory"); // 后面不再分配；这不是释放内存。
    }
    __syncthreads();
    const uint32_t taddr = tmem_addr_slot;

    // (2) gA 按 [m][k]，gB 按 [n][k] 存放；二者均为 K-major。
    // swz128 返回字节偏移，除 sizeof(bf16) 后才能作为数组下标。
    for (int i = tid; i < M * K; i += blockDim.x)
        sA[swz128(i / K, (i % K) * 2) / 2] = gA[i];
    for (int i = tid; i < N * K; i += blockDim.x)
        sB[swz128(i / K, (i % K) * 2) / 2] = gB[i];

    // (3) 每个写入线程发布自己的 generic shared stores，再集合。
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    // (4) 单线程发射四次 K=16，覆盖总 K=64。
    if (tid == 0) {
        const uint32_t a_base = static_cast<uint32_t>(__cvta_generic_to_shared(sA));
        const uint32_t b_base = static_cast<uint32_t>(__cvta_generic_to_shared(sB));
        // idesc: D=F32(1), A/B=BF16(1), M/16, N/8。
        // transpose=0 表示 K-major；dense、negate 等位均为 0。
        constexpr uint32_t idesc = (1u << 4) | (1u << 7) | (1u << 10)
                                | ((N >> 3) << 17) | ((M >> 4) << 24);
        for (int k0 = 0; k0 < K; k0 += 16) {
            // 8 行 * 每行 128B = SBO 1024B；K-major swizzle 不使用 LBO。
            const uint64_t adesc = make_desc_sm100(a_base + k0 * 2, 0, 1024, 2);
            const uint64_t bdesc = make_desc_sm100(b_base + k0 * 2, 0, 1024, 2);
            // 第一次禁用旧 D：无需先清零 TMEM；其后三次累加。
            asm volatile(
                "{ .reg .pred accumulate;\n"
                "  setp.ne.u32 accumulate, %4, 0;\n"
                "  tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, accumulate;\n"
                "}"
                :: "r"(taddr), "l"(adesc), "l"(bdesc), "r"(idesc), "r"(k0)
                : "memory");
        }
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
                     :: "r"(mbar) : "memory");
    }

    // (5) init 后首轮 phase=0。commit 不阻塞；所有消费者等待其完成通知。
    mbar_wait(mbar, 0);
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    // (6) 为便于学习，一次只读一列（x1）；每个线程得到一个 FP32。
    // warp 提供它的 32 行区域起点，指令把其中的行分给 lane 0..31。
    // 高 16 位是 TMEM 行，低 16 位是列；同 warp 传入相同地址。
    for (int n = 0; n < N; ++n) {
        const uint32_t read_addr = taddr + ((warp * 32) << 16) + n;
        uint32_t bits;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];"
                     : "=r"(bits) : "r"(read_addr) : "memory");
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        gD[(warp * 32 + lane) * N + n] = __uint_as_float(bits);
    }

    // (7) 等所有 warp 读完再释放。dealloc 直接接收 TMEM 地址和列数。
    __syncthreads();
    if (warp == 0)
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                     :: "r"(taddr), "r"(t_cols) : "memory");
    if (tid == 0)
        asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"(mbar) : "memory");
}

int main(int argc, char** argv) {
    unsigned seed = argc > 1 ? (unsigned)atoi(argv[1]) : 42;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(-3, 3);
    std::vector<__nv_bfloat16> hA(M * K), hB(N * K);
    std::vector<float> ref(M * N, 0.f);
    for (auto& v : hA) v = __float2bfloat16((float)dist(rng));
    for (auto& v : hB) v = __float2bfloat16((float)dist(rng));
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
            for (int k = 0; k < K; k++)
                ref[m * N + n] += __bfloat162float(hA[m * K + k]) *
                                  __bfloat162float(hB[n * K + k]);
    __nv_bfloat16 *dA, *dB;
    float* dD;
    CUDA_CHECK(cudaMalloc(&dA, M * K * 2));
    CUDA_CHECK(cudaMalloc(&dB, N * K * 2));
    CUDA_CHECK(cudaMalloc(&dD, M * N * 4));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), M * K * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), N * K * 2, cudaMemcpyHostToDevice));
    tcgen05_tile<<<1, 128>>>(dA, dB, dD);
    CUDA_CHECK_KERNEL();
    std::vector<float> got(M * N);
    CUDA_CHECK(cudaMemcpy(got.data(), dD, M * N * 4, cudaMemcpyDeviceToHost));
    long bad = 0;
    for (int i = 0; i < M * N; i++)
        if (got[i] != ref[i]) {
            if (bad < 5)
                printf("MISMATCH D[%d][%d]: got %.1f want %.1f\n", i / N,
                       i % N, got[i], ref[i]);
            bad++;
        }
    if (bad) printf("FAIL seed=%u: %ld / %d\n", seed, bad, M * N);
    else printf("PASS seed=%u: %d / %d\n", seed, M * N, M * N);
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dD));
    return bad != 0;
}
