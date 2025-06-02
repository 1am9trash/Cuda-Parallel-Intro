#ifndef GPU_INTERFACE_HPP
#define GPU_INTERFACE_HPP


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