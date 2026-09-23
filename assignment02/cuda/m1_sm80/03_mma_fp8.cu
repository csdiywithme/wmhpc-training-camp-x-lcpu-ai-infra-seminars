#include <cuda_fp8.h>
#include <charconv>
#include <cstring>
#include <limits>
#include <random>
#include "../common.h"

using fp8 = __nv_fp8_e4m3;
#include "fragment_map.cuh"

constexpr int M = 16;
constexpr int N = 8;
constexpr int K = 32;

// 1.3：host 脚手架由助教提供；kernel 由学员编写，助教按请求修正映射复用及打包。
// A[M][K]、B[K][N]、D[M][N] 均按行优先存储。
// 启动配置固定为一个 block、32 个线程；计算 D = A * B，初始累加值为 0。
__global__ void mma_fp8(const fp8* A, const fp8* B, float* D) {
    // 手工装载 FP8 fragment；不使用 ldmatrix。
    int lane = threadIdx.x;
    // 16 * 32 / 32 = 16, 32 * 8 / 32 = 8
    fp8 a[16], b[8];
    for(int i = 0; i < 16; ++i) a[i] = A[a_row_of(lane, i) * K + a_col_of(lane, i)];
    for(int i = 0; i < 8; ++i) b[i] = B[b_row_of(lane, i) * N + b_col_of(lane, i)];
    unsigned ra[4], rb[2];

    static_assert(sizeof(unsigned) == 4 && sizeof(fp8) == 1);
    // 复制 FP8 编码，不做数值转换，也不通过未对齐的 unsigned* 读取。
    for (int i = 0; i < 4; ++i) memcpy(&ra[i], &a[i * 4], sizeof(ra[i]));
    for (int i = 0; i < 2; ++i) memcpy(&rb[i], &b[i * 4], sizeof(rb[i]));
    float c[4] = {0.f, 0.f, 0.f, 0.f}, d[4];
    asm volatile(
    "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
    : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
    : "r"(ra[0]), "r"(ra[1]), "r"(ra[2]), "r"(ra[3]), "r"(rb[0]),
        "r"(rb[1]), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
    for(int i = 0; i < 4; ++i) D[d_row_of(lane, i) * N + d_col_of(lane, i)] = d[i];
}

int main(int argc, char** argv) {
    unsigned int seed = 0;
    if (argc != 2) {
        printf("FAIL: usage: %s <unsigned-seed>\n", argv[0]);
        return 1;
    }
    const char* end = argv[1] + std::strlen(argv[1]);
    const auto parsed = std::from_chars(argv[1], end, seed);
    if (parsed.ec != std::errc{} || parsed.ptr != end) {
        printf("FAIL: seed must be an unsigned integer\n");
        return 1;
    }

    fp8 hA[M * K];
    fp8 hB[K * N];
    float ref[M * N] = {};
    float got[M * N];

    // [-4, 4] 中的小整数可由 E4M3 精确表示；乘积和累加也可由 FP32 精确表示。
    // 使用 mt19937 原始输出取模，使相同 seed 的输入可复现。
    std::mt19937 rng(seed);
    for (auto& a : hA) a = fp8(static_cast<float>(static_cast<int>(rng() % 9) - 4));
    for (auto& b : hB) b = fp8(static_cast<float>(static_cast<int>(rng() % 9) - 4));

    // 从转换后的 FP8 输入计算 reference，避免参考值与 GPU 的实际输入不一致。
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n)
            for (int k = 0; k < K; ++k)
                ref[m * N + n] += static_cast<float>(hA[m * K + k]) *
                                  static_cast<float>(hB[k * N + n]);

    // NaN 用于发现漏写的输出元素；不能把初始输出当成 MMA 的零累加器。
    for (auto& value : got) value = std::numeric_limits<float>::quiet_NaN();

    fp8 *dA = nullptr, *dB = nullptr;
    float* dD = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, sizeof(hA)));
    CUDA_CHECK(cudaMalloc(&dB, sizeof(hB)));
    CUDA_CHECK(cudaMalloc(&dD, sizeof(got)));
    CUDA_CHECK(cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dD, got, sizeof(got), cudaMemcpyHostToDevice));

    mma_fp8<<<1, 32>>>(dA, dB, dD);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(got, dD, sizeof(got), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dD));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dA));

    int bad = 0;
    for (int i = 0; i < M * N; ++i)
        if (got[i] != ref[i]) ++bad;  // 严格相等检查；NaN 也会判为不匹配。

    if (bad == 0) {
        printf("PASS seed=%u: %d / %d elements matched\n", seed, M * N, M * N);
        return 0;
    }
    printf("FAIL seed=%u: %d / %d mismatches\n", seed, bad, M * N);
    int shown = 0;
    for (int i = 0; i < M * N && shown < 8; ++i) {
        if (got[i] != ref[i]) {
            printf("MISMATCH D[%d][%d]: got %.9g, want %.9g\n",
                   i / N, i % N, got[i], ref[i]);
            ++shown;
        }
    }
    return 1;
}
