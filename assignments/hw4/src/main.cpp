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
    constexpr unsigned int n = 4096000;
    constexpr unsigned int n_gpu = 1;
    std::vector<unsigned int> block_sizes = {32, 64, 128, 256, 512, 1024};
    std::vector<float> a = create_vector(n, 1.0, 0.0), b = create_vector(n, 1.0, 0.0);
    double cpu_sum, gpu_sum;

    std::cout << "------------------------\n";
    profile_on_cpu(a, b, cpu_sum, n, warmup_iter, test_iter);
    for (auto block_size : block_sizes) {
        profile_on_gpu(a.data(), b.data(), &gpu_sum, n, n_gpu, block_size, warmup_iter, test_iter);
    }
    std::cout << "\nResult Correctness: " << ((fabs((cpu_sum - gpu_sum) / cpu_sum) < 1e-6)? "True" : "False") << "\n\n";
    std::cout << "------------------------\n";

    return 0;
}
