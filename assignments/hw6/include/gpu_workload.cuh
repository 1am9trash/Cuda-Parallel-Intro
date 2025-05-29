#ifndef GPU_WORKLOAD_CUH
#define GPU_WORKLOAD_CUH

#include <cuda_runtime.h>


__global__ void histogram_gmem_kernel(
    const float *a,
    unsigned int *hist,
    const unsigned int n_bin,
    const float bin_size,
    unsigned int n
);

__global__ void histogram_smem_kernel(
    const float *a,
    unsigned int *hist,
    const unsigned int n_bin,
    const float bin_size,
    unsigned int n
);

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
);

void profile_on_gpu(
    const float *a,
    unsigned int *hist,
    const unsigned int n_bin,
    const float bin_size,
    const unsigned int n, 
    const unsigned int block_size,
    const unsigned int warmup_iter, const unsigned int test_iter
);

#endif