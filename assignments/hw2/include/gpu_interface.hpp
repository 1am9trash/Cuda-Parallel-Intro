#ifndef GPU_INTERFACE_HPP
#define GPU_INTERFACE_HPP


void profile_on_gpu(
    const float *a,
    double *sum,
    const unsigned int n, 
    const unsigned int block_size,
    const unsigned int warmup_iter, const unsigned int test_iter
);

#endif