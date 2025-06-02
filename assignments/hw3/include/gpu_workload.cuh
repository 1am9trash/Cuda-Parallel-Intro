#ifndef GPU_WORKLOAD_CUH
#define GPU_WORKLOAD_CUH

#include <cuda_runtime.h>


__global__ void poisson3d_kernel(
    double *grid_old,
    double *grid_new,
    double *grid_rho,
    double *error,
    unsigned int n
);

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
);

void profile_on_gpu(
    double *grid0,
    double *grid1,
    double *grid_rho,
    const unsigned int n,
    const double h,
    const unsigned int block_size_x,
    const unsigned int block_size_y,
    const unsigned int block_size_z
);

#endif