#ifndef CPU_WORKLOAD_HPP
#define CPU_WORKLOAD_HPP

#include <vector>


void calculate_on_cpu(
    const std::vector<float> &a, 
    double &sum,
    unsigned int n
);

void profile_on_cpu(
    const std::vector<float> &a, 
    double &sum,
    const unsigned int n, 
    const unsigned int warmup_iter, const unsigned int test_iter
);

#endif