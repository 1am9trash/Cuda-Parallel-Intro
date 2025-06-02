#ifndef CPU_WORKLOAD_HPP
#define CPU_WORKLOAD_HPP

#include <vector>


unsigned int calculate_on_cpu(
    std::vector<double> &grid0, 
    std::vector<double> &grid1, 
    std::vector<double> &grid_rho, 
    unsigned int n
);

void profile_on_cpu(
    std::vector<double> &grid0, 
    std::vector<double> &grid1, 
    std::vector<double> &grid_rho, 
    const unsigned int n,
    const double h
);

#endif