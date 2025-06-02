#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include "utils.hpp"
#include "cpu_workload.hpp"


unsigned int calculate_on_cpu(
    std::vector<double> &grid0, 
    std::vector<double> &grid1, 
    std::vector<double> &grid_rho, 
    unsigned int n
) {
    double error, diff, eps = 1e-10;
    unsigned int iter = 0;
    std::vector<double> *grid_old = &grid0;
    std::vector<double> *grid_new = &grid1;
    unsigned int position_id;

    do {
        error = 0.0;
        for (unsigned int i = 1; i < n - 1; i++) {
            for (unsigned int j = 1; j < n - 1; j++) {
                for (unsigned int k = 1; k < n - 1; k++) {
                    position_id = i * n * n + j * n + k;

                    (*grid_new)[position_id] = (
                        (*grid_old)[position_id - 1] 
                        + (*grid_old)[position_id + 1]
                        + (*grid_old)[position_id - n]
                        + (*grid_old)[position_id + n] 
                        + (*grid_old)[position_id - n * n]
                        + (*grid_old)[position_id + n * n]
                        + grid_rho[position_id]
                    ) / 6;

                    diff = (*grid_new)[position_id] - (*grid_old)[position_id];
                    error += diff * diff;
                }
            }
        }

        std::swap(grid_old, grid_new);
        iter++;
        error = sqrt(error);
    } while (error > eps);

    return iter;
}

void profile_on_cpu(
    std::vector<double> &grid0, 
    std::vector<double> &grid1, 
    std::vector<double> &grid_rho, 
    const unsigned int n,
    const double h
) {
    CPUTimer timer = CPUTimer();
    unsigned int iter = 0;

    initialize_grid(grid0.data(), grid1.data(), grid_rho.data(), n, h);
    timer.start_timer();
    iter = calculate_on_cpu(grid0, grid1, grid_rho, n);
    timer.stop_timer();

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\nCPU:\n";
    std::cout << "n = " << n << "\n";
    std::cout << "iter = " << iter << "\n";
    std::cout << "Total Time: " << timer.get_elaspsed_time_in_ms() << " ms\n";
}