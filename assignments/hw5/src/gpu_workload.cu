#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>
#include "utils.hpp"
#include "gpu_utils.cuh"

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>((float *)pointer)[0])


__global__ void heat_diffusion_kernel(
    float *grid_old,
    const float *grid_old_left, const float *grid_old_right, const float *grid_old_top, const float *grid_old_bottom,
    float *grid_new,
    double *error
) {
    extern __shared__ double cache[];

    unsigned int slice_x = blockDim.x * gridDim.x;
    unsigned int slice_y = blockDim.y * gridDim.y;
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int cache_id = threadIdx.x + threadIdx.y * blockDim.x;

    unsigned int position_id = x + y * slice_x;
    float left, right, top, bottom;
    int edge = 0;
    double diff = 0.0;

    if (x > 0 && x < slice_x - 1) {
        left = grid_old[position_id - 1];
        right = grid_old[position_id + 1];
    } else if (x == 0) {
        if (grid_old_left == NULL) {
            edge = 1;
        } else {
            left = grid_old_left[(slice_x - 1) + y * slice_x];
            right = grid_old[position_id + 1];
        }
    } else if (x == slice_x - 1) {
        if (grid_old_right == NULL) {
            edge = 1;
        } else {
            left = grid_old[position_id - 1];
            right = grid_old_right[y * slice_x];
        }
    }

    if (y > 0 && y < slice_y - 1) {
        top = grid_old[position_id - slice_x];
        bottom = grid_old[position_id + slice_x];
    } else if (y == 0) {
        if (grid_old_top == NULL) {
            edge = 1;
        } else {
            top = grid_old_top[x + (slice_y - 1) * slice_x];
            bottom = grid_old[position_id + slice_x];
        }
    } else if (y == slice_y - 1) {
        if (grid_old_bottom == NULL) {
            edge = 1;
        } else {
            top = grid_old[position_id - slice_x];
            bottom = grid_old_bottom[x];
        }
    }

    if (edge == 0) {
        grid_new[position_id] = 0.25 * (left + right + top + bottom);
        diff = grid_new[position_id] - grid_old[position_id];
    }

    cache[cache_id] = diff * diff;
    __syncthreads();

    unsigned int id = blockDim.x * blockDim.y / 2;
    while (id > 0) {
        if (cache_id < id) {
            cache[cache_id] += cache[cache_id + id];
        }
        __syncthreads();
        id /= 2;
    }

    unsigned int block_id = blockIdx.x + blockIdx.y * gridDim.x;
    if (cache_id == 0)
        error[block_id] = cache[0];
}

unsigned int kernel_runner(
    float *grid0,
    float *grid1,
    const unsigned int n,
    const unsigned int n_gpu_x,
    const unsigned int n_gpu_y,
    const unsigned int block_size_x,
    const unsigned int block_size_y,
    const unsigned int grid_size_x,
    const unsigned int grid_size_y,
    float *time_log
) {
    GPUTimer h2d_timer = GPUTimer();
    GPUTimer compute_timer = GPUTimer();
    GPUTimer d2h_timer = GPUTimer();

    // ------ Host to Device IO phase ------

    float **device_grid0;
    float **device_grid1;
    double **device_error;
    double *host_error;
    const unsigned int byte_count = sizeof(float) * n * n;
    const unsigned int error_byte_count = sizeof(double) * grid_size_x * grid_size_y;

    unsigned int n_gpu = n_gpu_x * n_gpu_y;
    device_grid0 = (float **)malloc(n_gpu * sizeof(float *));
    device_grid1 = (float **)malloc(n_gpu * sizeof(float *));
    device_error = (double **)malloc(n_gpu * sizeof(double *));
    host_error = (double *)malloc(error_byte_count);

    unsigned int slice_x = n / n_gpu_x;
    unsigned int slice_y = n / n_gpu_y;

    omp_set_num_threads(n_gpu);
    #pragma omp parallel
    {
        int cpu_thread_id = omp_get_thread_num();
        cudaSetDevice(cpu_thread_id);
        unsigned int cpu_thread_id_x = cpu_thread_id % n_gpu_x;
        unsigned int cpu_thread_id_y = cpu_thread_id / n_gpu_x;

        unsigned int cpu_thread_id_left = ((cpu_thread_id_x + n_gpu_x - 1) % n_gpu_x) + cpu_thread_id_y * n_gpu_x;
        unsigned int cpu_thread_id_right = ((cpu_thread_id_x + 1) % n_gpu_x) + cpu_thread_id_y * n_gpu_x;
        unsigned int cpu_thread_id_top = cpu_thread_id_x + ((cpu_thread_id_y + n_gpu_y - 1) % n_gpu_y) * n_gpu_x;
        unsigned int cpu_thread_id_bottom = cpu_thread_id_x + ((cpu_thread_id_y + 1) % n_gpu_y) * n_gpu_x;
        cudaDeviceEnablePeerAccess(cpu_thread_id_left, 0);
        cudaDeviceEnablePeerAccess(cpu_thread_id_right, 0);
        cudaDeviceEnablePeerAccess(cpu_thread_id_top, 0);
        cudaDeviceEnablePeerAccess(cpu_thread_id_bottom, 0);

        if (cpu_thread_id == 0) {
            h2d_timer.start_timer();
        }

        cudaMalloc((void **)&device_grid0[cpu_thread_id], byte_count / n_gpu);
        cudaMalloc((void **)&device_grid1[cpu_thread_id], byte_count / n_gpu);
        cudaMalloc((void **)&device_error[cpu_thread_id], error_byte_count / n_gpu);

        for (unsigned int i = 0; i < slice_y; i++) {
            float *h, *d;
            h = grid0 + cpu_thread_id_x * slice_x + (cpu_thread_id_y * slice_y + i) * n;
            d = device_grid0[cpu_thread_id] + i * slice_x;
            cudaMemcpy(d, h, slice_x * sizeof(float), cudaMemcpyHostToDevice);
            h = grid1 + cpu_thread_id_x * slice_x + (cpu_thread_id_y * slice_y + i) * n;
            d = device_grid1[cpu_thread_id] + i * slice_x;
            cudaMemcpy(d, h, slice_x * sizeof(float), cudaMemcpyHostToDevice);
        }

    #pragma omp barrier

        if (cpu_thread_id == 0) {
            h2d_timer.stop_timer();
        }
    }

    // ------ compute phase ------

    compute_timer.start_timer();

    double error, eps = 1e-10;
    unsigned int iter = 0;
    float **device_grid_new = device_grid0;
    float **device_grid_old = device_grid1;

    dim3 block(block_size_x, block_size_y);
    dim3 grid(grid_size_x / n_gpu_x, grid_size_y / n_gpu_y);

    do {
        #pragma omp parallel
        {
            int cpu_thread_id = omp_get_thread_num();
            cudaSetDevice(cpu_thread_id);
            unsigned int cpu_thread_id_x = cpu_thread_id % n_gpu_x;
            unsigned int cpu_thread_id_y = cpu_thread_id / n_gpu_x;

            float *grid_old_left = (cpu_thread_id_x == 0) ? NULL : device_grid_old[cpu_thread_id_x - 1 + cpu_thread_id_y * n_gpu_x];
            float *grid_old_right = (cpu_thread_id_x == n_gpu_x - 1) ? NULL : device_grid_old[cpu_thread_id_x + 1 + cpu_thread_id_y * n_gpu_x];
            float *grid_old_top = (cpu_thread_id_y == 0) ? NULL : device_grid_old[cpu_thread_id_x + (cpu_thread_id_y - 1) * n_gpu_x];
            float *grid_old_bottom = (cpu_thread_id_y == n_gpu_y - 1) ? NULL : device_grid_old[cpu_thread_id_x + (cpu_thread_id_y + 1) * n_gpu_x];
            heat_diffusion_kernel<<<grid, block, block_size_x * block_size_y * sizeof(double)>>>(
                device_grid_old[cpu_thread_id],
                grid_old_left, grid_old_right, grid_old_top, grid_old_bottom,
                device_grid_new[cpu_thread_id],
                device_error[cpu_thread_id]
            );
            cudaDeviceSynchronize();

            cudaMemcpy(
                host_error + grid_size_x * grid_size_y / n_gpu * cpu_thread_id, 
                device_error[cpu_thread_id], 
                error_byte_count / n_gpu, 
                cudaMemcpyDeviceToHost
            );
        }

        error = 0.0;
        for (unsigned int i = 0; i < grid_size_x * grid_size_y; i++)
            error += host_error[i];
        error = sqrt(error);
        iter++;

        std::swap(device_grid_new, device_grid_old);
    } while (error > eps);

    compute_timer.stop_timer();

    // ------ Device to Host IO phase ------

    d2h_timer.start_timer();

    #pragma omp parallel
    {
        int cpu_thread_id = omp_get_thread_num();
        cudaSetDevice(cpu_thread_id);
        unsigned int cpu_thread_id_x = cpu_thread_id % n_gpu_x;
        unsigned int cpu_thread_id_y = cpu_thread_id / n_gpu_x;

        for (unsigned int i = 0; i < slice_y; i++) {
            float *h, *d;
            h = grid0 + cpu_thread_id_x * slice_x + (cpu_thread_id_y * slice_y + i) * n;
            d = device_grid_old[cpu_thread_id] + i * slice_x;
            cudaMemcpy(h, d, slice_x * sizeof(float), cudaMemcpyDeviceToHost);
        }

        cudaFree(device_grid0[cpu_thread_id]);
        cudaFree(device_grid1[cpu_thread_id]);
        cudaFree(device_error[cpu_thread_id]);
    }

    free(device_grid0);
    free(device_grid1);
    free(device_error);
    free(host_error);

    d2h_timer.stop_timer();

    // ------ Logging results ------

    if (time_log) {
        printf("Iter: %d, Error: %.12f\n", iter, error);
        time_log[0] += h2d_timer.get_elaspsed_time_in_ms();
        time_log[1] += compute_timer.get_elaspsed_time_in_ms();
        time_log[2] += d2h_timer.get_elaspsed_time_in_ms();
    }

    return iter;
}

void profile_on_gpu(
    float *grid0,
    float *grid1,
    const unsigned int n, 
    const unsigned int n_gpu_x,
    const unsigned int n_gpu_y,
    const unsigned int block_size_x,
    const unsigned int block_size_y
) {
    assert(n % n_gpu_x == 0);
    assert(n % n_gpu_y == 0);
    assert(n % block_size_x == 0);
    assert(n % block_size_y == 0);
    assert((n / block_size_x) % n_gpu_x == 0);
    assert((n / block_size_y) % n_gpu_y == 0);

    float time_log[3] = {0.0, 0.0, 0.0};
    unsigned int grid_size_x = n / block_size_x;
    unsigned int grid_size_y = n / block_size_y;

    initialize_grid(grid0, grid1, n);
    unsigned int iter = kernel_runner(grid0, grid1, n, n_gpu_x, n_gpu_y, block_size_x, block_size_y, grid_size_x, grid_size_y, &time_log[0]);

    printf("\nGPU:\n");
    printf("n = %d\n", n);
    printf("n_gpu_x = %d, n_gpu_y = %d\n", n_gpu_x, n_gpu_y);
    printf("block_size_x = %d, block_size_y = %d\n", block_size_x, block_size_y);
    printf("grid_size_x = %d, grid_size_y = %d\n", grid_size_x, grid_size_y);
    printf("iter: %d\n", iter);
    printf("Host to Device IO: %.6f\n", time_log[0]);
    printf("Compute: %.6f\n", time_log[1]);
    printf("Device to Host IO: %.6f\n", time_log[2]);
    printf("Total Time: %.6f\n\n", (time_log[0] + time_log[1] + time_log[2]));
}