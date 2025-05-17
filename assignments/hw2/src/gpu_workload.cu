#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "utils.hpp"
#include "gpu_utils.cuh"

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>((float *)pointer)[0])


__global__ void matrix_trace_reduce_kernel(
    const float *a,
    float *sum,
    unsigned int n
) {
    extern __shared__ float cache[];

    unsigned int cache_id = threadIdx.x;
    unsigned int cur_id = blockIdx.x * blockDim.x + threadIdx.x;

    float tmp = 0.0;
    while (cur_id < n) {
        tmp += a[cur_id * n + cur_id];
        cur_id += blockDim.x * gridDim.x; 
    }
    cache[cache_id] = tmp;
    __syncthreads();

    unsigned int reduce_id = blockDim.x >> 1;;
    while (reduce_id != 0) {
        if(cache_id < reduce_id)
            cache[cache_id] += cache[cache_id + reduce_id]; 
        __syncthreads();
        reduce_id >>= 1;
    }
    
    if (cache_id == 0)
        sum[blockIdx.x] = cache[0];
}

void kernel_runner(
    const float *a,
    double *sum,
    const unsigned int n,
    const unsigned int block_size,
    const unsigned int grid_size,
    float *time_log
) {
    GPUTimer h2d_timer = GPUTimer();
    GPUTimer gpu_compute_timer = GPUTimer();
    GPUTimer d2h_timer = GPUTimer();
    CPUTimer cpu_compute_timer = CPUTimer();

    float *device_a;
    float *host_sum;
    float *device_sum;
    const unsigned int a_byte_count = sizeof(float) * n * n;
    const unsigned int sum_byte_count = sizeof(float) * grid_size;

    const dim3 block(block_size);
    const dim3 grid(grid_size);

    // allocate and move data to GPU
    h2d_timer.start_timer();
    cudaMalloc((void **)&device_a, a_byte_count);
    host_sum = (float *)malloc(sum_byte_count);
    cudaMalloc((void **)&device_sum, sum_byte_count);
    cudaMemcpy(device_a, a, a_byte_count, cudaMemcpyHostToDevice);
    h2d_timer.stop_timer();

    // gpu compute
    gpu_compute_timer.start_timer();
    matrix_trace_reduce_kernel<<<grid, block, sizeof(float) * block_size>>>(device_a, device_sum, n);
    gpu_compute_timer.stop_timer();

    // deallcocate, move data to CPU
    d2h_timer.start_timer();
    cudaMemcpy(host_sum, device_sum, sum_byte_count, cudaMemcpyDeviceToHost);
    cudaFree(device_a);
    cudaFree(device_sum);
    d2h_timer.stop_timer();

    // gpu compute
    cpu_compute_timer.start_timer();
    *sum = 0.0;
    for (size_t i = 0; i < grid_size; i++) {
        *sum += (double)host_sum[i];
    }
    free(host_sum);
    cpu_compute_timer.stop_timer();

    if (time_log) {
        time_log[0] += h2d_timer.get_elaspsed_time_in_ms();
        time_log[1] += gpu_compute_timer.get_elaspsed_time_in_ms();
        time_log[2] += d2h_timer.get_elaspsed_time_in_ms();
        time_log[3] += cpu_compute_timer.get_elaspsed_time_in_ms();
    }
}

void profile_on_gpu(
    const float *a,
    double *sum,
    const unsigned int n, 
    const unsigned int block_size,
    const unsigned int warmup_iter, const unsigned int test_iter
) {
    unsigned int grid_sizes[3] = {
        (n + block_size - 1) / block_size, 
        (n + block_size - 1) / block_size / 2,
        (n + block_size - 1) / block_size / 4,
    };

    for (size_t i = 0; i < 3; i++) {
        float time_log[4] = {0.0, 0.0, 0.0, 0.0};
        unsigned int grid_size = grid_sizes[i];

        for (size_t _ = 0; _ < warmup_iter; _++) {
            kernel_runner(a, sum, n, block_size, grid_size, NULL);
        }
        for (size_t _ = 0; _ < test_iter; _++) {
            kernel_runner(a, sum, n, block_size, grid_size, &time_log[0]);
        }

        printf("\nGPU:\n");
        printf("\nResult: %.6f\n", *sum);
        printf("n = %d, block_size = %d, grid_size = %d\n", n, block_size, grid_size);
        printf("Host to Device IO: %.6f\n", time_log[0] / test_iter);
        printf("GPU Compute: %.6f\n", time_log[1] / test_iter);
        printf("Device to Host IO: %.6f\n", time_log[2] / test_iter);
        printf("CPU Compute: %.6f\n", time_log[3] / test_iter);
        printf("Total Time: %.6f\n", (time_log[0] + time_log[1] + time_log[2] + time_log[3]) / test_iter);
    }
}
