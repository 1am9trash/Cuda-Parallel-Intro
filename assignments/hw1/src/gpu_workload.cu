#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "gpu_utils.cuh"

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>((float *)pointer)[0])


__global__ void add_matrix_kernel(
    const float *a,
    const float *b,
    float *c,
    unsigned int n
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n * n) {
        c[index] = 1 / a[index] + 1 / b[index];
    }
}

__global__ void vectorized_add_matrix_kernel(
    const float *a,
    const float *b,
    float *c,
    unsigned int n
) {
    const unsigned int index = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (index < n * n) {
        float4 reg_a = FETCH_FLOAT4(&a[index]);
        float4 reg_b = FETCH_FLOAT4(&b[index]);
        float4 reg_c = FETCH_FLOAT4(&c[index]);
        reg_c.x = 1 / reg_a.x + 1 / reg_b.x;
        reg_c.y = 1 / reg_a.y + 1 / reg_b.y;
        reg_c.z = 1 / reg_a.z + 1 / reg_b.z;
        reg_c.w = 1 / reg_a.w + 1 / reg_b.w;
        FETCH_FLOAT4(&c[index]) = reg_c;
    }
}

void kernel_runner(
    const float *a,
    const float *b,
    float *c,
    const unsigned int n,
    const int vectorize,
    const unsigned int block_size,
    float *time_log
) {
    GPUTimer h2d_timer = GPUTimer();
    GPUTimer compute_timer = GPUTimer();
    GPUTimer d2h_timer = GPUTimer();

    float *device_a, *device_b, *device_c;
    const unsigned int byte_count = sizeof(float) * n * n;

    const int compute_per_thread = (vectorize == 0) ? 1 : 4;
    const dim3 block(block_size);
    const dim3 grid((n * n + compute_per_thread * block_size - 1) / (compute_per_thread * block_size));

    // move to GPU
    h2d_timer.start_timer();
    cudaMalloc((void **)&device_a, byte_count);
    cudaMalloc((void **)&device_b, byte_count);
    cudaMalloc((void **)&device_c, byte_count);
    cudaMemcpy(device_a, a, byte_count, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, b, byte_count, cudaMemcpyHostToDevice);
    h2d_timer.stop_timer();

    // compute
    compute_timer.start_timer();
    if (vectorize == 0) {
        add_matrix_kernel<<<grid, block>>>(device_a, device_b, device_c, n);
    } else {
        vectorized_add_matrix_kernel<<<grid, block>>>(device_a, device_b, device_c, n);
    }
    compute_timer.stop_timer();

    // move to cpu
    d2h_timer.start_timer();
    cudaMemcpy(c, device_c, byte_count, cudaMemcpyDeviceToHost);
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);
    d2h_timer.stop_timer();

    if (time_log) {
        time_log[0] += h2d_timer.get_elaspsed_time_in_ms();
        time_log[1] += compute_timer.get_elaspsed_time_in_ms();
        time_log[2] += d2h_timer.get_elaspsed_time_in_ms();
    }
}

void profile_on_gpu(
    const float *a,
    const float *b,
    float *c,
    const unsigned int n, 
    const unsigned int block_size,
    const unsigned int warmup_iter, const unsigned int test_iter
) {
    float time_log[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    for (size_t i = 0; i < warmup_iter; i++) {
        kernel_runner(a, b, c, n, 0, block_size, NULL);
    }

    for (size_t i = 0; i < test_iter; i++) {
        kernel_runner(a, b, c, n, 0, block_size, time_log);
    }

    for (size_t i = 0; i < test_iter; i++) {
        kernel_runner(a, b, c, n, 1, block_size, &time_log[3]);
    }

    printf("\nGPU:\n");
    printf("n = %d, block_size = %d\n", n, block_size);
    printf("\nScalar add:\n");
    printf("Host to Device IO: %.3f\n", time_log[0] / test_iter);
    printf("Compute: %.3f\n", time_log[1] / test_iter);
    printf("Device to Host IO: %.3f\n", time_log[2] / test_iter);
    printf("Total Time: %.3f\n", (time_log[0] + time_log[1] + time_log[2]) / test_iter);
    printf("\nVectorized add:\n");
    printf("Host to Device IO: %.3f\n", time_log[3] / test_iter);
    printf("Compute: %.3f\n", time_log[4] / test_iter);
    printf("Device to Host IO: %.3f\n", time_log[5] / test_iter);
    printf("Total Time: %.3f\n", (time_log[3] + time_log[4] + time_log[5]) / test_iter);
}