#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <numeric>

#define CHECK_CUDA(call)                                            \
    do {                                                             \
        cudaError_t err = call;                                      \
        if (err != cudaSuccess) {                                    \
            std::cerr << "CUDA error: "                              \
                      << cudaGetErrorString(err)                     \
                      << " at line " << __LINE__ << std::endl;       \
            exit(EXIT_FAILURE);                                      \
        }                                                            \
    } while (0)

__global__ void empty_kernel()
{
    // intentionally empty
}

double benchmark_kernel_launch(int iterations)
{
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Warmup
    for (int i = 0; i < 1000; i++) {
        empty_kernel<<<1,1>>>();
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < iterations; i++) {
        empty_kernel<<<1,1>>>();
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    // ms -> us
    return (ms * 1000.0) / iterations;
}

double benchmark_memcpy_h2d(
    void* h_ptr,
    void* d_ptr,
    size_t bytes,
    int iterations)
{
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Warmup
    for (int i = 0; i < 100; i++) {
        CHECK_CUDA(cudaMemcpy(
            d_ptr,
            h_ptr,
            bytes,
            cudaMemcpyHostToDevice));
    }

    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < iterations; i++) {
        CHECK_CUDA(cudaMemcpy(
            d_ptr,
            h_ptr,
            bytes,
            cudaMemcpyHostToDevice));
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return (ms * 1000.0) / iterations;
}

double benchmark_memcpy_d2h(
    void* h_ptr,
    void* d_ptr,
    size_t bytes,
    int iterations)
{
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Warmup
    for (int i = 0; i < 100; i++) {
        CHECK_CUDA(cudaMemcpy(
            h_ptr,
            d_ptr,
            bytes,
            cudaMemcpyDeviceToHost));
    }

    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < iterations; i++) {
        CHECK_CUDA(cudaMemcpy(
            h_ptr,
            d_ptr,
            bytes,
            cudaMemcpyDeviceToHost));
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return (ms * 1000.0) / iterations;
}

int main()
{
    const int iterations = 100;
    const size_t bytes = 1024 * 1024 * 256; // tiny transfer to expose startup latency

    std::cout << "CUDA Latency Benchmark\n";
    std::cout << "Iterations: " << iterations << "\n\n";

    // Device buffer
    void* d_ptr = nullptr;
    CHECK_CUDA(cudaMalloc(&d_ptr, bytes));

    // Pageable host memory
    void* h_pageable = malloc(bytes);

    // Pinned host memory
    void* h_pinned = nullptr;
    CHECK_CUDA(cudaMallocHost(&h_pinned, bytes));

    // ------------------------------------------------------------
    // Kernel launch latency
    // ------------------------------------------------------------

    double kernel_us = benchmark_kernel_launch(iterations);

    std::cout << "Kernel launch latency:\n";
    std::cout << "  " << kernel_us << " us\n\n";

    // ------------------------------------------------------------
    // Pageable memory transfer latency
    // ------------------------------------------------------------

    double pageable_h2d = benchmark_memcpy_h2d(
        h_pageable,
        d_ptr,
        bytes,
        iterations);

    double pageable_d2h = benchmark_memcpy_d2h(
        h_pageable,
        d_ptr,
        bytes,
        iterations);

    std::cout << "Pageable memcpy latency (" << bytes << " bytes):\n";
    std::cout << "  H2D: " << pageable_h2d << " us\n";
    std::cout << "  D2H: " << pageable_d2h << " us\n\n";

    // ------------------------------------------------------------
    // Pinned memory transfer latency
    // ------------------------------------------------------------

    double pinned_h2d = benchmark_memcpy_h2d(
        h_pinned,
        d_ptr,
        bytes,
        iterations);

    double pinned_d2h = benchmark_memcpy_d2h(
        h_pinned,
        d_ptr,
        bytes,
        iterations);

    std::cout << "Pinned memcpy latency (" << bytes << " bytes):\n";
    std::cout << "  H2D: " << pinned_h2d << " us\n";
    std::cout << "  D2H: " << pinned_d2h << " us\n";

    // Cleanup
    CHECK_CUDA(cudaFree(d_ptr));
    CHECK_CUDA(cudaFreeHost(h_pinned));
    free(h_pageable);

    return 0;
}