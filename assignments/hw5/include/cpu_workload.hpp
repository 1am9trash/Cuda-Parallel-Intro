#ifndef CPU_WORKLOAD_HPP
#define CPU_WORKLOAD_HPP

#include <vector>


unsigned int calculate_on_cpu(
    std::vector<float> &grid0, 
    std::vector<float> &grid1, 
    unsigned int n
);

void profile_on_cpu(
    std::vector<float> &grid0, 
    std::vector<float> &grid1, 
    const unsigned int n
);

#endif