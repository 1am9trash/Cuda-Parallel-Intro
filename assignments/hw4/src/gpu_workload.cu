#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>
#include "utils.hpp"
#include "gpu_utils.cuh"

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>((float *)pointer)[0])


__global__ void dot_product_kernel(
    const float *a,
    const float *b,
    float *sum,
    unsigned int n
) {
    extern __shared__ float cache[];

    unsigned int cache_id = threadIdx.x;
    unsigned int cur_id = blockIdx.x * blockDim.x + threadIdx.x;

    float tmp = 0.0;
    while (cur_id < n) {
        tmp += a[cur_id] * b[cur_id];
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
    const float *b,
    double *sum,
    const unsigned int n,
    const unsigned int n_gpu,
    const unsigned int block_size,
    const unsigned int grid_size,
    float *time_log
) {
    GPUTimer h2d_timer = GPUTimer();
    GPUTimer gpu_compute_timer = GPUTimer();
    GPUTimer d2h_timer = GPUTimer();
    CPUTimer cpu_compute_timer = CPUTimer();

    float *device_a;
    float *device_b;
    float *host_sum;
    float *device_sum;
    const unsigned int byte_count = sizeof(float) * n;
    const unsigned int sum_byte_count = sizeof(float) * grid_size;

    host_sum = (float *)malloc(sum_byte_count);

    const dim3 block(block_size);
    const dim3 grid(grid_size / n_gpu);

    omp_set_num_threads(n_gpu);
    #pragma omp parallel private(device_a, device_b, device_sum)
    {

        int cpu_thread_id = omp_get_thread_num();
        cudaSetDevice(cpu_thread_id);

        if (cpu_thread_id == 0) {
            h2d_timer.start_timer();
        }

        // allocate and move data to GPU
        unsigned int offset = n / n_gpu * cpu_thread_id;
        cudaMalloc((void **)&device_a, byte_count / n_gpu);
        cudaMalloc((void **)&device_b, byte_count / n_gpu);
        cudaMalloc((void **)&device_sum, sum_byte_count / n_gpu);
        cudaMemcpy(device_a, a + offset, byte_count / n_gpu, cudaMemcpyHostToDevice);
        cudaMemcpy(device_b, b + offset, byte_count / n_gpu, cudaMemcpyHostToDevice);
        
    #pragma omp barrier

        if (cpu_thread_id == 0) {
            h2d_timer.stop_timer();
            gpu_compute_timer.start_timer();
        }

        // gpu compute
        unsigned int sum_offset = grid_size / n_gpu * cpu_thread_id;
        dot_product_kernel<<<grid, block, sizeof(float) * block_size>>>(device_a, device_b, device_sum, n / n_gpu);

    #pragma omp barrier

        if (cpu_thread_id == 0) {
            gpu_compute_timer.stop_timer();
            d2h_timer.start_timer();
        }

        // deallcocate, move data to CPU
        cudaMemcpy(host_sum + sum_offset, device_sum, sum_byte_count / n_gpu, cudaMemcpyDeviceToHost);
        cudaFree(device_a);
        cudaFree(device_b);
        cudaFree(device_sum);

    #pragma omp barrier

        if (cpu_thread_id == 0) {
            d2h_timer.stop_timer();
        }
    }

    cpu_compute_timer.start_timer();

    // gpu compute
    *sum = 0.0;
    for (size_t i = 0; i < grid_size; i++) {
        *sum += (double)host_sum[i];
    }

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
    const float *b,
    double *sum,
    const unsigned int n, 
    const unsigned int n_gpu,
    const unsigned int block_size,
    const unsigned int warmup_iter, const unsigned int test_iter
) {
    unsigned int grid_sizes[3] = {
        (n + block_size - 1) / block_size, 
        (n + block_size - 1) / block_size / 2,
        (n + block_size - 2) / block_size / 4,
    };

    assert(n % n_gpu == 0);
    assert(block_size % n_gpu == 0);
    for (size_t i = 0; i < 3; i++) {
        float time_log[4] = {0.0, 0.0, 0.0, 0.0};
        unsigned int grid_size = grid_sizes[i];

        assert(grid_size % n_gpu == 0);

        for (size_t _ = 0; _ < warmup_iter; _++) {
            kernel_runner(a, b, sum, n, n_gpu, block_size, grid_size, NULL);
        }
        for (size_t _ = 0; _ < test_iter; _++) {
            kernel_runner(a, b, sum, n, n_gpu, block_size, grid_size, &time_log[0]);
        }

        printf("\nGPU:\n");
        printf("\nResult: %.6f\n", *sum);
        printf("n = %d, n_gpu = %d, block_size = %d, grid_size = %d\n", n, n_gpu, block_size, grid_size);
        printf("Host to Device IO: %.6f\n", time_log[0] / test_iter);
        printf("GPU Compute: %.6f\n", time_log[1] / test_iter);
        printf("Device to Host IO: %.6f\n", time_log[2] / test_iter);
        printf("CPU Compute: %.6f\n", time_log[3] / test_iter);
        printf("Total Time: %.6f\n", (time_log[0] + time_log[1] + time_log[2] + time_log[3]) / test_iter);
    }
}
