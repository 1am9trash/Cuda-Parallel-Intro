#ifndef GPU_WORKLOAD_CUH
#define GPU_WORKLOAD_CUH

#include <cuda_runtime.h>


__global__ void heat_diffusion_kernel(
    float *grid_old,
    const float *grid_old_left, const float *grid_old_right, const float *grid_old_top, const float *grid_old_bottom,
    float *grid_new,
    double *error
);

void kernel_runner(
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
);

void profile_on_gpu(
    float *grid0,
    float *grid1,
    const unsigned int n, 
    const unsigned int n_gpu_x,
    const unsigned int n_gpu_y,
    const unsigned int block_size_x,
    const unsigned int block_size_y
); 

#endif