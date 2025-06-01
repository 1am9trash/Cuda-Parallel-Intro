#include <iostream>
#include <vector>
#include <iomanip>
#include <string>
#include <cmath>
#include "utils.hpp"
#include "cpu_workload.hpp"
#include "gpu_interface.hpp"


int main() {
    constexpr unsigned int n = 1024;
    std::vector<std::vector<unsigned int>> n_gpus = {{1, 1}, {1, 2}, {2, 1}};
    std::vector<std::vector<unsigned int>> block_sizes = {{4, 4}, {8, 8}, {16, 16}, {32, 32}};
    std::vector<float> grid0(n * n), grid1(n * n);

    std::cout << "------------------------\n";
    profile_on_cpu(grid0, grid1, n);
    save_grid_to_file(grid0, n, "result_log/cpu_grid.data");

    for (auto n_gpu : n_gpus) {
        for (auto block_size : block_sizes) {
            profile_on_gpu(grid0.data(), grid1.data(), n, n_gpu[0], n_gpu[1], block_size[0], block_size[1]);

            std::ostringstream oss;
            oss << "result_log/gpu_grid_gpu" << n_gpu[0] << "x" << n_gpu[1] << "_bs" << block_size[0] << "x" << block_size[1] << ".data";
            std::string fname = oss.str();
            save_grid_to_file(grid0, n, fname);
        }
    }
    std::cout << "------------------------\n";

    return 0;
}