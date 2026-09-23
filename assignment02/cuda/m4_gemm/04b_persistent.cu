// 4.4 experiment B: one CTA processes multiple output tiles.
// Based on the unmodified 4.3 pipeline; TMEM allocation persists across tiles.
// 问题 4.3(FROM-SCRATCH,模块压轴):多级缓冲流水。
//
// 从你自己的 02_tma.cu 出发,把单缓冲扩成 STAGES 级循环缓冲:TMA 往
// 前预取后续 K 段,mma 消费当前段,装载与计算重叠。STAGES 是编译参数:
//   STAGES=4 make -B run/m4_gemm/03_pipeline
// (-B 不能省:只改 -D 不改文件,make 会认为无需重编。)
//
// 明确不要求:warp specialization、persistent kernel、epilogue 融合。
// 不设达成率门槛,评分看实验与归因质量。
//
// 两个已知事实,直接告知:
//   1. smem 用量 = STAGES*(BM+BN)*BK*2,STAGES>=3 起超过 48KB 静态
//      上限,必须动态 smem + cudaFuncSetAttribute(main 已配好)。
//   2. 一条真实的流水线 hazard(我们开发答案时踩到的,写出来让你避开):
//      "机会式预取"(try_wait 非阻塞,空了就发)不能替代"强制发射"。
//      若本轮要消费的那段 TMA 在早先检查时 stage 未空而被跳过,后面
//      wait full 等的就是一条从未发出的拷贝——死锁。症状签名很典型:
//      1024^3 侥幸全过,4096^3 必挂(13 万次机会必中一次)。正确结构:
//      本轮要消费的 TMA 用阻塞等 empty 保证发出,机会式 try_wait 只
//      用于更深的预取。另外 empty mbarrier 必须每 stage 一个:单个
//      mbar 的 parity 区分不了相隔 2 轮的完成,STAGES>=2 必然歧义。
//
// 交付:
//   - 梯子表第三行(4096^3,默认 STAGES=3)
//   - stages 扫描表:S ∈ {2,3,4,6},在两个形状上各扫一遍——4096^3 与
//     M=256 N=4096 K=16384(小 grid、长 K)。两张表的 S 敏感度不一样,
//     解释差异来自什么(提示方向:每 SM 常驻 block 数怎么随 smem 用量
//     变、块间并发本身能隐藏多少延迟)。./sweep_stages.sh 会跑全表
//   - 流水时空图:任选一个 S,画出稳态下 TMA/mma 在各 stage 上的重叠
//   - handout 4.3 的三问:瓶颈移动;梯子表逐级归因(含 assignment01
//     的 naive matmul 同口径对照);smem 与 TMEM 谁先顶住扩 stage/tile
//
// 运行:make run/m4_gemm/03_pipeline;./bin/m4_gemm/03_pipeline M N K
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

constexpr int BM = 128, BN = 64, BK = 64;
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
            : "=r"(done)
            : "r"(mbar), "r"(phase)
            : "memory");
}

// 非阻塞版:成功返回 true。机会式深预取用它。
__device__ inline bool mbar_try(uint32_t mbar, uint32_t phase) {
    uint32_t done;
    asm volatile(
        "{\n.reg .pred p;\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
        "selp.b32 %0, 1, 0, p;\n}"
        : "=r"(done)
        : "r"(mbar), "r"(phase)
        : "memory");
    return done;
}

__global__ void gemm_pipeline(const __nv_bfloat16* gA, const __nv_bfloat16* gB,
                              float* gD, int M, int N, int K,
                              const __grid_constant__ CUtensorMap tmapA,
                              const __grid_constant__ CUtensorMap tmapB) {
    extern __shared__ uint8_t smem_raw[];
    uint8_t* smem =
        (uint8_t*)(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023);

    // TODO:把你 4.2 的 kernel 扩成 NSTAGE 级流水。参考结构:
    // (1) smem 划成 NSTAGE 段,stage s 的 A/B 起点自己排;mbarrier 每
    //     stage 两个:full[s](TMA 到达)、empty[s](mma 消费完成)

    // (3) 主循环 it:
    //     - 强制发射:若第 it 轮 TMA 还没发,阻塞等 empty[it%NSTAGE]
    //       后补发(见文件头 hazard;empty 的 parity 按该 stage 被复用
    //       的轮次算,第一次复用等的是上一轮使用的完成)
    //     - 机会式深预取:try_wait 下一个待发 stage 的 empty,成功就
    //       继续发,失败立刻停,不许阻塞
    //     - 等 full[it%NSTAGE](parity = (it/NSTAGE)&1)→ tcgen05.fence
    //       → mma(与 4.2 相同,累加位口径不变)→ commit 到
    //       empty[it%NSTAGE]
    // (4) drain:等最后一轮 mma 的 empty 到达,再进 epilogue
    // 初始化 full（TMA 完成）、empty（MMA 完成）及 TMEM。
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid % 32;

    __shared__ uint32_t tmem_addr_slot;
    __shared__ __align__(8) uint64_t barrier[NSTAGE];
    __shared__ __align__(8) uint64_t barrier_full[NSTAGE];
    const uint32_t slot_saddr = static_cast<uint32_t>(__cvta_generic_to_shared(&tmem_addr_slot));
    constexpr uint32_t t_cols = BN; // M = 128, FP32
    if (warp == 0) {
        // 输入是 shared 地址；返回的 TMEM 地址被写到 shared 变量里。
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                     :: "r"(slot_saddr), "r"(t_cols) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"
                     ::: "memory"); // 后面不再分配；这不是释放内存。
    }
    __syncthreads();
    const uint32_t taddr = tmem_addr_slot;
    const int tile_count = (M / BM) * (N / BN);
    for (int tile_id = blockIdx.x; tile_id < tile_count; tile_id += gridDim.x) {
        // Fresh barrier phases for each output tile; keep TMEM allocated
        // across tiles so this CTA can process several tiles persistently.
        if (tid == 0) {
            for (int stage = 0; stage < NSTAGE; ++stage) {
                const uint32_t e = __cvta_generic_to_shared(&barrier[stage]);
                const uint32_t f = __cvta_generic_to_shared(&barrier_full[stage]);
                asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                             :: "r"(e) : "memory");
                asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                             :: "r"(f) : "memory");
            }
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
        __syncthreads();

    // 本 block 的输出 tile。
    const int tileM = (tile_id / (N / BN)) * BM;
    const int tileN = (tile_id % (N / BN)) * BN;
    const int sa_ofst = 0;
    const int sb_ofst = BM*BK*2;
    // A/B 在 shared memory 中的偏移单位为字节。

    // (2) 预热:先发 min(NSTAGE, iters) 轮 TMA(发第 it 轮 = 对 stage
    //     it%NSTAGE 做 arrive.expect_tx + 两条 cp.async.bulk.tensor)
    const int iters = K / BK;
    const int warm_n = min(NSTAGE, iters);
    int next_to_issue = warm_n;
    for (int i = 0; i < warm_n; i ++) {
        int s = i;
        const uint32_t mbar_full = static_cast<uint32_t>(__cvta_generic_to_shared(&barrier_full[s]));

        const int smem_stage_ofst = s * (BM + BN) * BK * 2;
        __nv_bfloat16* sA = reinterpret_cast<__nv_bfloat16*>(smem + smem_stage_ofst + sa_ofst);
        __nv_bfloat16* sB = reinterpret_cast<__nv_bfloat16*>(smem + smem_stage_ofst + sb_ofst);

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
            const int kBase = i * BK;

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
    }
    for (int it = 0; it < iters; it += 1){
        int s = it % NSTAGE;
        const uint32_t mbar = static_cast<uint32_t>(__cvta_generic_to_shared(&barrier[s]));
        const uint32_t mbar_full = static_cast<uint32_t>(__cvta_generic_to_shared(&barrier_full[s]));

        const int smem_stage_ofst = s * (BM + BN) * BK * 2;
        __nv_bfloat16* sA = reinterpret_cast<__nv_bfloat16*>(smem + smem_stage_ofst + sa_ofst);
        __nv_bfloat16* sB = reinterpret_cast<__nv_bfloat16*>(smem + smem_stage_ofst + sb_ofst);

        // 预取尚未发出当前 tile 时，阻塞等待并保证发出。
        if (tid == 0 && next_to_issue == it) {
            const uint32_t bytes = 2 * BK * (BM + BN);
            const int previous_it = it - NSTAGE;
            mbar_wait(mbar, (previous_it / NSTAGE) & 1);
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
            ++next_to_issue;
        }

        // 等 A/B 搬运完成，再交给 MMA。
        mbar_wait(mbar_full, (it / NSTAGE) & 1);
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
            // 空闲就预取下一块；未空闲便留给后续迭代或兜底路径。
            while (next_to_issue < iters) {
                const int pre_s = next_to_issue % NSTAGE;
                const uint32_t pre_mbar = static_cast<uint32_t>(__cvta_generic_to_shared(&barrier[pre_s]));
                const int previous_it = next_to_issue - NSTAGE;
                if (!mbar_try(pre_mbar, (previous_it / NSTAGE) & 1))
                    break;
                const uint32_t pre_mbar_full = static_cast<uint32_t>(__cvta_generic_to_shared(&barrier_full[pre_s]));
                const uint32_t bytes = 2 * BK * (BM + BN);

                const int psmem_stage_ofst = pre_s * (BM + BN) * BK * 2;
                __nv_bfloat16* psA = reinterpret_cast<__nv_bfloat16*>(smem + psmem_stage_ofst + sa_ofst);
                __nv_bfloat16* psB = reinterpret_cast<__nv_bfloat16*>(smem + psmem_stage_ofst + sb_ofst);
                asm volatile(
                    "{\n"
                    "  .reg .b64 state;\n"
                    "  mbarrier.arrive.expect_tx.shared::cta.b64 "
                    "state, [%0], %1;\n"
                    "}\n"
                    :
                    : "r"(pre_mbar_full), "r"(bytes)
                    : "memory"
                );
                const uint32_t pdstA = static_cast<uint32_t>(__cvta_generic_to_shared(psA));
                const uint32_t pdstB = static_cast<uint32_t>(__cvta_generic_to_shared(psB));
                const int kBase = next_to_issue * BK;

                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cluster.global"
                    ".mbarrier::complete_tx::bytes "
                    "[%0], [%1, {%2, %3}], [%4];"
                    :
                    : "r"(pdstA),
                    "l"(&tmapA),
                    "r"(kBase),
                    "r"(tileM),
                    "r"(pre_mbar_full)
                    : "memory"
                );
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cluster.global"
                    ".mbarrier::complete_tx::bytes "
                    "[%0], [%1, {%2, %3}], [%4];"
                    :
                    : "r"(pdstB),
                    "l"(&tmapB),
                    "r"(kBase),
                    "r"(tileN),
                    "r"(pre_mbar_full)
                    : "memory"
                );
                ++next_to_issue;
            }
        }
    }
    // This CTA will reinitialize every barrier for the next output tile.
    // Drain the last use of *each* stage before invalidating them.
    for (int stage = 0; stage < NSTAGE; ++stage) {
        const int distance = (iters - 1 - stage + NSTAGE) % NSTAGE;
        const int final_it = iters - 1 - distance;
        if (final_it >= 0) {
            const uint32_t e = __cvta_generic_to_shared(&barrier[stage]);
            mbar_wait(e, (final_it / NSTAGE) & 1);
        }
    }
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    // 从 TMEM 读回并写入 D，global 行跨度为 N。
    for (int n = 0; n < BN; ++n) {
        const uint32_t read_addr = taddr + ((warp * 32) << 16) + n;
        uint32_t bits;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];"
                     : "=r"(bits) : "r"(read_addr) : "memory");
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        gD[tileM*N + tileN + (warp * 32 + lane) * N + n] = __uint_as_float(bits);
    }

    // All four warps have finished reading the current tile's TMEM values.
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();
    if (tid == 0) {
        for (int stage = 0; stage < NSTAGE; ++stage) {
            const uint32_t e = __cvta_generic_to_shared(&barrier[stage]);
            const uint32_t f = __cvta_generic_to_shared(&barrier_full[stage]);
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
    const int capacity = device.multiProcessorCount * 3; // S=3 smem limit
    dim3 grid(total_tiles < capacity ? total_tiles : capacity);
    // NSTAGE=3 时 72KB+对齐余量,超 48KB 静态上限,动态 smem 必须。
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
    printf("[4.4 persistent S=%d] M=%d N=%d K=%d  %s(bad=%ld)  %.2f ms  %.1f "
           "TFLOPS  (cuBLAS %.1f, 达成率 %.0f%%)\n",
           NSTAGE, M, N, K, bad ? "FAIL" : "PASS", bad, ms, tflops,
           cub_tflops, 100.0 * tflops / cub_tflops);
    cublasDestroy(h);
    return bad != 0;
}
