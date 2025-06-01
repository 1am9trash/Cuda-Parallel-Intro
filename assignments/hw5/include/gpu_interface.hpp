#ifndef GPU_INTERFACE_HPP
#define GPU_INTERFACE_HPP


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