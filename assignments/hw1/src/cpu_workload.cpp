#include <vector>
#include <iostream>
#include <iomanip>
#include "utils.hpp"
#include "cpu_workload.hpp"


void calculate_on_cpu(
    const std::vector<float> &a, 
    const std::vector<float> &b, 
    std::vector<float> &c, 
    unsigned int n
) {
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            const unsigned int index = i * n + j;
            c[index] = 1 / a[index] + 1 / b[index];
        }
    }
}

void profile_on_cpu(
    const std::vector<float> &a, 
    const std::vector<float> &b, 
    std::vector<float> &c, 
    const unsigned int n, 
    const unsigned int warmup_iter, const unsigned int test_iter
) {
    CPUTimer timer = CPUTimer();

    for (size_t i = 0; i < warmup_iter; i++) {
        calculate_on_cpu(a, b, c, n);
    }
    timer.start_timer();
    for (size_t i = 0; i < test_iter; i++) {
        calculate_on_cpu(a, b, c, n);
    }
    timer.stop_timer();

    float ms = timer.get_elaspsed_time_in_ms() / test_iter;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\nCPU:\n";
    std::cout << "n = " << n << "\n";
    std::cout << "Total Time: " << ms << " ms\n";
}