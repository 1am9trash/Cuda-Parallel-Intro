#include <iostream>
#include <vector>
#include <iomanip>
#include "utils.hpp"
#include "cpu_workload.hpp"
#include "gpu_interface.hpp"


int main() {
    constexpr unsigned int warmup_iter = 20;
    constexpr unsigned int test_iter = 30;
    constexpr unsigned int n = 1024;
    std::vector<unsigned int> block_sizes = {128, 256, 512, 1024};

    std::vector<float> a = create_matrix(n, n, 1.0, 0.0);
    std::vector<float> b = create_matrix(n, n, 1.0, 0.0);
    std::vector<float> cpu_c(n * n), gpu_c(n * n);

    std::cout << "------------------------\n";
    profile_on_cpu(a, b, cpu_c, n, warmup_iter, test_iter);
    for (auto block_size : block_sizes) {
        profile_on_gpu(a.data(), b.data(), gpu_c.data(), n, block_size, warmup_iter, test_iter);
    }
    std::cout << "\nResult Correctness: " << (check_matrix_same(cpu_c, gpu_c) == 1? "True" : "False") << "\n\n";
    std::cout << "------------------------\n";

    return 0;
}