#ifndef GPU_INTERFACE_HPP
#define GPU_INTERFACE_HPP


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