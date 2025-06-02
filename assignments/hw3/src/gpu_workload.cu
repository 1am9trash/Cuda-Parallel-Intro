#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>
#include "utils.hpp"
#include "gpu_utils.cuh"

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>((float *)pointer)[0])


__global__ void poisson3d_kernel(
    double *grid_old,
    double *grid_new,
    double *grid_rho,
    double *error,
    unsigned int n
) {
    extern __shared__ double cache[];

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;
    unsigned int cache_id = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    double diff = 0.0;

    if (
        i > 0 && i < n - 1
        && j > 0 && j < n - 1
        && k > 0 && k < n - 1
    ) {
        unsigned int position_id = i + j * n + k * n * n;

        grid_new[position_id] = (
            grid_old[position_id - 1]
            + grid_old[position_id + 1]
            + grid_old[position_id - n]
            + grid_old[position_id + n]
            + grid_old[position_id - n * n]
            + grid_old[position_id + n * n]
            + grid_rho[position_id]
        ) / 6;

        diff = grid_new[position_id] - grid_old[position_id];
    }

    cache[cache_id] = diff * diff;
    __syncthreads();

    unsigned int id = blockDim.x * blockDim.y * blockDim.z / 2;
    while (id > 0) {
        if (cache_id < id) {
            cache[cache_id] += cache[cache_id + id];
        }
        __syncthreads();
        id /= 2;
    }

    unsigned int block_id = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    if (cache_id == 0)
        error[block_id] = cache[0];
}

unsigned int kernel_runner(
    double *grid0,
    double *grid1,
    double *grid_rho,
    const unsigned int n,
    const unsigned int block_size_x,
    const unsigned int block_size_y,
    const unsigned int block_size_z,
    const unsigned int grid_size_x,
    const unsigned int grid_size_y,
    const unsigned int grid_size_z,
    float *time_log
) {
    GPUTimer h2d_timer = GPUTimer();
    GPUTimer compute_timer = GPUTimer();
    GPUTimer d2h_timer = GPUTimer();

    // ------ Host to Device IO phase ------

    double *device_grid0;
    double *device_grid1;
    double *device_grid_rho;
    double *device_error;
    double *host_error;
    const unsigned int byte_count = sizeof(double) * n * n * n;
    const unsigned int error_byte_count = sizeof(double) * grid_size_x * grid_size_y * grid_size_z;

    // allocate and move data to GPU
    h2d_timer.start_timer();
    cudaMalloc((void **)&device_grid0, byte_count);
    cudaMalloc((void **)&device_grid1, byte_count);
    cudaMalloc((void **)&device_grid_rho, byte_count);
    cudaMalloc((void **)&device_error, error_byte_count);
    host_error = (double *)malloc(error_byte_count);
    cudaMemcpy(device_grid0, grid0, byte_count, cudaMemcpyHostToDevice);
    cudaMemcpy(device_grid1, grid1, byte_count, cudaMemcpyHostToDevice);
    cudaMemcpy(device_grid_rho, grid_rho, byte_count, cudaMemcpyHostToDevice);
    h2d_timer.stop_timer();

    // compute
    compute_timer.start_timer();
    double error, eps = 1e-10;
    unsigned int iter = 0;
    double *device_grid_new = device_grid0;
    double *device_grid_old = device_grid1;

    dim3 block(block_size_x, block_size_y, block_size_z);
    dim3 grid(grid_size_x, grid_size_y, grid_size_z);

    do {
        poisson3d_kernel<<<grid, block, block_size_x * block_size_y * block_size_z * sizeof(double)>>>(
            device_grid_old,
            device_grid_new,
            device_grid_rho,
            device_error,
            n
        );

        cudaMemcpy(host_error, device_error, error_byte_count, cudaMemcpyDeviceToHost);

        error = 0.0;
        for (unsigned int i = 0; i < grid_size_x * grid_size_y * grid_size_z; i++)
            error += host_error[i];
        error = sqrt(error);
        iter++;

        std::swap(device_grid_new, device_grid_old);
    } while (error > eps);
    compute_timer.stop_timer();

    // deallcocate, move data to CPU
    d2h_timer.start_timer();
    cudaMemcpy(grid0, device_grid_old, byte_count, cudaMemcpyDeviceToHost);
    cudaFree(device_grid0);
    cudaFree(device_grid1);
    cudaFree(device_grid_rho);
    cudaFree(device_error);
    free(host_error);
    d2h_timer.stop_timer();

    if (time_log) {
        printf("Iter: %d, Error: %.12f\n", iter, error);
        time_log[0] += h2d_timer.get_elaspsed_time_in_ms();
        time_log[1] += compute_timer.get_elaspsed_time_in_ms();
        time_log[2] += d2h_timer.get_elaspsed_time_in_ms();
    }

    return iter;
}

void profile_on_gpu(
    double *grid0,
    double *grid1,
    double *grid_rho,
    const unsigned int n,
    const double h,
    const unsigned int block_size_x,
    const unsigned int block_size_y,
    const unsigned int block_size_z
) {
    float time_log[3] = {0.0, 0.0, 0.0};
    unsigned int grid_size_x = (n + block_size_x - 1) / block_size_x;
    unsigned int grid_size_y = (n + block_size_y - 1) / block_size_y;
    unsigned int grid_size_z = (n + block_size_z - 1) / block_size_z;

    initialize_grid(grid0, grid1, grid_rho, n, h);
    unsigned int iter = kernel_runner(
        grid0, grid1, grid_rho, n,
        block_size_x, block_size_y, block_size_z,
        grid_size_x, grid_size_y, grid_size_z,
        &time_log[0]
    );

    printf("\nGPU:\n");
    printf("n = %d\n", n);
    printf("block_size_x = %d, block_size_y = %d, block_size_z = %d\n", block_size_x, block_size_y, block_size_z);
    printf("grid_size_x = %d, grid_size_y = %d, grid_size_z = %d\n", grid_size_x, grid_size_y, grid_size_z);
    printf("iter: %d\n", iter);
    printf("Host to Device IO: %.6f\n", time_log[0]);
    printf("Compute: %.6f\n", time_log[1]);
    printf("Device to Host IO: %.6f\n", time_log[2]);
    printf("Total Time: %.6f\n\n", (time_log[0] + time_log[1] + time_log[2]));
}