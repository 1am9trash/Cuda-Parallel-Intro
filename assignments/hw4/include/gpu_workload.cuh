#ifndef GPU_WORKLOAD_CUH
#define GPU_WORKLOAD_CUH

#include <cuda_runtime.h>


__global__ void dot_product_kernel(
    const float *a,
    const float *b,
    float *sum,
    unsigned int n
);

void kernel_runner(
    const float *a,
    const float *b,
    double *sum,
    const unsigned int n,
    const unsigned int n_gpu,
    const unsigned int block_size,
    const unsigned int grid_size,
    float *time_log
);

void profile_on_gpu(
    const float *a,
    const float *b,
    double *sum,
    const unsigned int n, 
    const unsigned int n_gpu,
    const unsigned int block_size,
    const unsigned int warmup_iter, const unsigned int test_iter
); 

#endif