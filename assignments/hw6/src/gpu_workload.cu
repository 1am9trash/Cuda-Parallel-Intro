#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "utils.hpp"
#include "gpu_utils.cuh"

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>((float *)pointer)[0])


__global__ void histogram_gmem_kernel(
    const float *a,
    unsigned int *hist,
    const unsigned int n_bin,
    const float bin_size,
    unsigned int n
) {
    unsigned int cur_id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    while (cur_id < n) {
        unsigned int hist_id = a[cur_id] / bin_size;
        atomicAdd(&hist[hist_id], 1);
        cur_id += stride;
    }
    __syncthreads();
}

__global__ void histogram_smem_kernel(
    const float *a,
    unsigned int *hist,
    const unsigned int n_bin,
    const float bin_size,
    unsigned int n
) {
    extern __shared__ unsigned int hist_smem[];

    // assume block_size is larger than n_bin
    if (threadIdx.x < n_bin) {
        hist_smem[threadIdx.x] = 0;
    }
    __syncthreads();
    
    unsigned int cur_id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    while (cur_id < n) {
        unsigned int hist_id = a[cur_id] / bin_size;
        atomicAdd(&hist_smem[hist_id], 1);
        cur_id += stride;
    }
    __syncthreads();

    if (threadIdx.x < n_bin) {
        atomicAdd(&hist[threadIdx.x], hist_smem[threadIdx.x]);
    }
    __syncthreads();
}

__global__ void histogram_smem_vectorized_access_kernel(
    const float *a,
    unsigned int *hist,
    const unsigned int n_bin,
    const float bin_size,
    unsigned int n
) {
    extern __shared__ unsigned int hist_smem[];

    // assume block_size is larger than n_bin
    if (threadIdx.x < n_bin) {
        hist_smem[threadIdx.x] = 0;
    }
    __syncthreads();
    
    constexpr unsigned int compute_per_iter = 4;
    unsigned int cur_id = compute_per_iter * (blockIdx.x * blockDim.x + threadIdx.x);
    unsigned int stride = compute_per_iter * blockDim.x * gridDim.x;

    while (cur_id < n) {
        float4 data = FETCH_FLOAT4(a + cur_id);
        uint4 hist_id = make_uint4(
            data.x / bin_size,
            data.y / bin_size,
            data.z / bin_size,
            data.w / bin_size
        );
        atomicAdd(&hist_smem[hist_id.x], 1);
        atomicAdd(&hist_smem[hist_id.y], 1);
        atomicAdd(&hist_smem[hist_id.z], 1);
        atomicAdd(&hist_smem[hist_id.w], 1);
        cur_id += stride;
    }
    __syncthreads();

    if (threadIdx.x < n_bin) {
        atomicAdd(&hist[threadIdx.x], hist_smem[threadIdx.x]);
    }
    __syncthreads();
}

void kernel_runner(
    const float *a,
    unsigned int *hist,
    const unsigned int n_bin,
    const float bin_size,
    const unsigned int n,
    const unsigned int mode,
    const unsigned int block_size,
    const unsigned int grid_size,
    float *time_log
) {
    GPUTimer h2d_timer = GPUTimer();
    GPUTimer gpu_compute_timer = GPUTimer();
    GPUTimer d2h_timer = GPUTimer();

    float *device_a;
    unsigned int *device_hist;
    const unsigned int byte_count = sizeof(float) * n;
    const unsigned int hist_byte_count = sizeof(unsigned int) * n_bin;

    // allocate and move data to GPU
    h2d_timer.start_timer();
    cudaMalloc((void **)&device_a, byte_count);
    cudaMalloc((void **)&device_hist, hist_byte_count);
    cudaMemcpy(device_a, a, byte_count, cudaMemcpyHostToDevice);
    h2d_timer.stop_timer();
        
    // gpu compute
    gpu_compute_timer.start_timer();
    if (mode == 0) {
        const dim3 block(block_size);
        const dim3 grid(grid_size);
        histogram_gmem_kernel<<<grid, block>>>(device_a, device_hist, n_bin, bin_size, n);
    } else if (mode == 1) {
        assert(block_size >= n_bin);
        const dim3 block(block_size);
        const dim3 grid(grid_size);
        histogram_smem_kernel<<<grid, block, sizeof(unsigned int) * n_bin>>>(device_a, device_hist, n_bin, bin_size, n);
    } else if (mode == 2) {
        assert(block_size >= n_bin);
        assert(n % 4 == 0);
        assert(grid_size % 4 == 0);
        const dim3 block(block_size);
        const dim3 grid(grid_size / 4);
        histogram_smem_vectorized_access_kernel<<<grid, block, sizeof(unsigned int) * n_bin>>>(device_a, device_hist, n_bin, bin_size, n);
    }
    gpu_compute_timer.stop_timer();

    // deallcocate, move data to CPU
    d2h_timer.start_timer();
    cudaMemcpy(hist, device_hist, hist_byte_count, cudaMemcpyDeviceToHost);
    cudaFree(device_a);
    cudaFree(device_hist);
    d2h_timer.stop_timer();

    if (time_log) {
        time_log[0] += h2d_timer.get_elaspsed_time_in_ms();
        time_log[1] += gpu_compute_timer.get_elaspsed_time_in_ms();
        time_log[2] += d2h_timer.get_elaspsed_time_in_ms();
    }
}

void profile_on_gpu(
    const float *a,
    unsigned int *hist,
    const unsigned int n_bin,
    const float bin_size,
    const unsigned int n, 
    const unsigned int block_size,
    const unsigned int warmup_iter, const unsigned int test_iter
) {
    unsigned int grid_size = (n + block_size - 1) / block_size;

    for (unsigned int mode = 0; mode < 3; mode++) {
        float time_log[3] = {0.0, 0.0, 0.0};

        for (size_t i = 0; i < n_bin; i++) {
            hist[i] = 0;
        }

        for (size_t _ = 0; _ < warmup_iter; _++) {
            kernel_runner(a, hist, n_bin, bin_size, n, mode, block_size, grid_size, NULL);
        }

        for (size_t _ = 0; _ < test_iter; _++) {
            kernel_runner(a, hist, n_bin, bin_size, n, mode, block_size, grid_size, time_log);
        }

        printf("\nGPU:\n");
        printf("\nMode: %d\n", mode);
        printf("Result:\n");
        for (size_t i = 0; i < n_bin; i++) {
            if (i % 4 == 0)
                printf("\n");
            printf("bin[%2zu] = %-10u ", i, hist[i]);
        }
        printf("\n");
        printf("Host to Device IO: %.6f\n", time_log[0] / test_iter);
        printf("GPU Compute: %.6f\n", time_log[1] / test_iter);
        printf("Device to Host IO: %.6f\n", time_log[2] / test_iter);
        printf("Total Time: %.6f\n", (time_log[0] + time_log[1] + time_log[2]) / test_iter);
    }
}
