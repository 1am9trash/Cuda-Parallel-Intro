#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include "utils.hpp"
#include "cpu_workload.hpp"
#include "gpu_interface.hpp"


int main() {
    constexpr unsigned int warmup_iter = 20;
    constexpr unsigned int test_iter = 30;
    constexpr unsigned int n = 81920000;

    constexpr float range_mx = 20;
    constexpr unsigned int n_bin = 32;
    constexpr float bin_size = range_mx / n_bin;
    std::vector<unsigned int> block_sizes = {64, 128, 256, 512, 1024};

    // n element in [0, range_mx] with exp distribution
    // std::vector<float> a = create_vector_with_exp_distribution(n, range_mx);
    std::vector<float> a = create_vector(n, range_mx, 0);
    std::vector<unsigned int> cpu_hist(n_bin), gpu_hist(n_bin);

    std::cout << "------------------------\n";
    profile_on_cpu(a, cpu_hist, bin_size, n, warmup_iter, test_iter);
    for (auto block_size : block_sizes) {
        profile_on_gpu(a.data(), gpu_hist.data(), n_bin, bin_size, n, block_size, warmup_iter, test_iter);
    }

    bool is_correct = true;
    for (unsigned int i = 0; i < n_bin; ++i) {
        if (cpu_hist[i] != gpu_hist[i])
            is_correct = false;
    }
    std::cout << "\nResult Correctness: " << is_correct << "\n\n";
    std::cout << "------------------------\n";

    return 0;
}
