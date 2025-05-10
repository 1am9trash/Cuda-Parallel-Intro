#ifndef GPU_WORKLOAD_CUH
#define GPU_WORKLOAD_CUH

#include <cuda_runtime.h>


__global__ void add_matrix_kernel(
    const float *a,
    const float *b,
    float *c,
    unsigned int n
);

__global__ void vectorized_add_matrix_kernel(
    const float *a,
    const float *b,
    float *c,
    unsigned int n
);

void kernel_runner(
    const float *a,
    const float *b,
    float *c,
    const unsigned int n,
    const int vectorize, const dim3 block,
    float *time_log
);

void profile_on_gpu(
    const float *a,
    const float *b,
    float *c,
    const unsigned int n, 
    const unsigned int block_size,
    const unsigned int warmup_iter, const unsigned int test_iter
);

#endif