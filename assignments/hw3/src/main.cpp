#include <iostream>
#include <vector>
#include <iomanip>
#include <string>
#include <cmath>
#include "utils.hpp"
#include "cpu_workload.hpp"
#include "gpu_interface.hpp"


int main() {
    constexpr double h = 1.0;
    std::vector<unsigned int> ls = {8, 16, 32, 64};
    std::vector<unsigned int> block_size = {8, 8, 8};

    std::cout << "------------------------\n";
    for (auto l : ls) {
        unsigned int n3 = (l + 1) * (l + 1) * (l + 1); 
        std::vector<double> grid0(n3), grid1(n3), grid_rho(n3);

        profile_on_cpu(grid0, grid1, grid_rho, l + 1, h);
        save_grid_to_file(grid0, l + 1, "result_log/cpu_grid_" + std::to_string(l) + ".data");

        profile_on_gpu(
            grid0.data(), grid1.data(), grid_rho.data(), l + 1, h, 
            block_size[0], block_size[1], block_size[2]
        );
        std::ostringstream oss;
        oss << "result_log/gpu_grid_" << std::to_string(l) << "_bs" << block_size[0] << "x" << block_size[1] << "x" << block_size[2] << ".data";
        std::string fname = oss.str();
        save_grid_to_file(grid0, l + 1, fname);

        std::cout << "------------------------\n";
    }

    return 0;
}