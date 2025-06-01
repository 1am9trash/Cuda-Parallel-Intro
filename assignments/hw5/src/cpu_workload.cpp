#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include "utils.hpp"
#include "cpu_workload.hpp"


unsigned int calculate_on_cpu(
    std::vector<float> &grid0, 
    std::vector<float> &grid1, 
    unsigned int n
) {
    double error, diff, eps = 1e-10;
    unsigned int iter = 0;
    std::vector<float> *grid_old = &grid0;
    std::vector<float> *grid_new = &grid1;
    unsigned int position_id;

    do {
        error = 0.0;
        for (unsigned int i = 1; i < n - 1; i++) {
            for (unsigned int j = 1; j < n - 1; j++) {
                position_id = i * n + j;

                (*grid_new)[position_id] = 0.25f * (
                    (*grid_old)[position_id - 1] + 
                    (*grid_old)[position_id + 1] + 
                    (*grid_old)[position_id - n] + 
                    (*grid_old)[position_id + n]
                );

                diff = (*grid_new)[position_id] - (*grid_old)[position_id];
                error += diff * diff;
            }
        }

        std::swap(grid_old, grid_new);
        iter++;
        error = sqrt(error);
    } while (error > eps);

    return iter;
}

void profile_on_cpu(
    std::vector<float> &grid0, 
    std::vector<float> &grid1, 
    const unsigned int n
) {
    CPUTimer timer = CPUTimer();
    unsigned int iter = 0;

    initialize_grid(grid0.data(), grid1.data(), n);
    timer.start_timer();
    iter = calculate_on_cpu(grid0, grid1, n);
    timer.stop_timer();

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\nCPU:\n";
    std::cout << "n = " << n << "\n";
    std::cout << "iter = " << iter << "\n";
    std::cout << "Total Time: " << timer.get_elaspsed_time_in_ms() << " ms\n";
}