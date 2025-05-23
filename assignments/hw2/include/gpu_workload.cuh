#ifndef GPU_WORKLOAD_CUH
#define GPU_WORKLOAD_CUH

#include <cuda_runtime.h>


__global__ void matrix_trace_reduce_kernel(
    const float *a,
    float *sum,
    unsigned int n
);

void kernel_runner(
    const float *a,
    double *sum,
    const unsigned int n,
    const unsigned int block_size,
    const unsigned int grid_size,
    float *time_log
);

void profile_on_gpu(
    const float *a,
    double *sum,
    const unsigned int n, 
    const unsigned int block_size,
    const unsigned int warmup_iter, const unsigned int test_iter
); 

#endif