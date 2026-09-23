// Measure the unchanged Assignment01 Bonus naive kernel at the M4 ladder shape.
// Rename only the original host main; the included kernel and default BS=16
// remain exactly those in assignment01/cuda/bonus/matmul.cu.
#define main assignment01_original_main
#include "../../assignment01/cuda/bonus/matmul.cu"
#undef main

#include <vector>

int main() {
    constexpr int M = 4096, N = 4096, K = 4096;
    constexpr float input = 1.0f / 128.0f;
    constexpr float expected = K * input * input;
    const size_t count = static_cast<size_t>(M) * K;
    std::vector<float> hA(count, input), hB(count, input), hC(count);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, count * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dC, 0xFF, count * sizeof(float)));

    const dim3 threads(BS, BS);
    const dim3 blocks((N + BS - 1) / BS, (M + BS - 1) / BS);
    matmul_naive<<<blocks, threads>>>(dA, dB, dC, M, N, K);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, count * sizeof(float), cudaMemcpyDeviceToHost));
    size_t bad = 0;
    for (const float value : hC) bad += value != expected;
    if (bad) {
        fprintf(stderr, "naive FP32 4096^3 FAIL: bad=%zu, expected=%g, first=%g\n",
                bad, static_cast<double>(expected), static_cast<double>(hC[0]));
        return 1;
    }

    constexpr int reps = 2;
    GpuTimer timer;
    timer.start();
    for (int i = 0; i < reps; ++i)
        matmul_naive<<<blocks, threads>>>(dA, dB, dC, M, N, K);
    const float ms = timer.stop_ms() / reps;
    CUDA_CHECK_KERNEL();
    const double tflops = 2.0 * M * N * K / (ms * 1e9);
    printf("[A01 naive FP32] BS=%d M=%d N=%d K=%d PASS(bad=0) "
           "reps=%d %.3f ms %.4f TFLOPS\n", BS, M, N, K, reps, ms, tflops);

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    return 0;
}
