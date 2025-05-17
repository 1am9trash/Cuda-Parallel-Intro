#include <vector>
#include <iostream>
#include <iomanip>
#include "utils.hpp"
#include "cpu_workload.hpp"


void calculate_on_cpu(
    const std::vector<float> &a, 
    double &sum,
    unsigned int n
) {
    sum = 0.0;
    for (size_t i = 0; i < n; i++)
        sum += double(a[i * n + i]);
}

void profile_on_cpu(
    const std::vector<float> &a, 
    double &sum,
    const unsigned int n, 
    const unsigned int warmup_iter, const unsigned int test_iter
) {
    CPUTimer timer = CPUTimer();

    for (size_t i = 0; i < warmup_iter; i++) {
        calculate_on_cpu(a, sum, n);
    }
    timer.start_timer();
    for (size_t i = 0; i < test_iter; i++) {
        calculate_on_cpu(a, sum, n);
    }
    timer.stop_timer();

    float ms = timer.get_elaspsed_time_in_ms() / test_iter;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\nCPU:\n";
    std::cout << "n = " << n << "\n";
    std::cout << "Total Time: " << ms << " ms\n";
}
