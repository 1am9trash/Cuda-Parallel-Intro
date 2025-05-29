#include <vector>
#include <iostream>
#include <iomanip>
#include "utils.hpp"
#include "cpu_workload.hpp"


void calculate_on_cpu(
    const std::vector<float> &a, 
    std::vector<unsigned int> &hist,
    const float bin_size,
    const unsigned int n
) {
    for (size_t i = 0; i < hist.size(); i++)
        hist[i] = 0;
    for (size_t i = 0; i < n; i++) {
        unsigned int bin_idx = a[i] / bin_size;
        hist[bin_idx]++;
    }
}

void profile_on_cpu(
    const std::vector<float> &a, 
    std::vector<unsigned int> &hist,
    const float bin_size,
    const unsigned int n, 
    const unsigned int warmup_iter, const unsigned int test_iter
) {
    CPUTimer timer = CPUTimer();

    for (size_t i = 0; i < warmup_iter; i++) {
        calculate_on_cpu(a, hist, bin_size, n);
    }
    timer.start_timer();
    for (size_t i = 0; i < test_iter; i++) {
        calculate_on_cpu(a, hist, bin_size, n);
    }
    timer.stop_timer();

    float ms = timer.get_elaspsed_time_in_ms() / test_iter;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\nCPU:\n";
    std::cout << "\nResult:";
    for (size_t i = 0; i < hist.size(); i++) {
        if (i % 4 == 0)
            std::cout << "\n";
        std::cout << "bin[" << std::setw(2) << std::right << i << "] = ";
        std::cout << std::setw(10) << std::left << hist[i] << " ";
    }
    std::cout << "\n\n";
    std::cout << "n = " << n << "\n";
    std::cout << "Total Time: " << ms << " ms\n";
}
