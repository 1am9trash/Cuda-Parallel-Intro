#include <vector>
#include <fstream>
#include <string>
#include <random>
#include <iostream>
#include <iomanip>
#include <ctime>
#include <algorithm>
#include "utils.hpp"


#define RAND_SEED 1337

void randomize_vector(std::vector<float> &vec, float mx, float mn) {
    static std::default_random_engine rng(RAND_SEED);
    std::uniform_real_distribution<float> dist(mn, mx);
    std::generate(vec.begin(), vec.end(), [&]{ return dist(rng); });
}

std::vector<float> create_matrix(unsigned int m, unsigned int n, float mx, float mn) {
    std::vector<float> mat(m * n);
    randomize_vector(mat, mx, mn);
    return mat;
}

std::vector<float> create_vector(unsigned int m, float mx, float mn) {
    std::vector<float> vec(m);
    randomize_vector(vec, mx, mn);
    return vec;
}

bool check_matrix_same(std::vector<float> &a, std::vector<float> &b) {
    constexpr float eps = 1e-6;

    if (a.size() != b.size()) {
        std::cerr << "Matrix a and b may have the same size.\n\n";
        return false;
    }

    for (size_t i = 0; i < a.size(); i++) {
        if (fabs(a[i] - b[i]) > eps) {
            return false;
        }
    }

    return true;
}

void print_matrix(std::vector<float> mat, unsigned int limit) {
    for (size_t i = 0; i < std::min((unsigned int)mat.size(), limit) ; i++) {
        std::cout << mat[i] << " ";
    }
    std::cout << "\n\n";
}

void initialize_grid(float *grid_new, float *grid_old, unsigned int n) {
    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < n; j++) {
            grid_new[i * n + j] = 0.0;
            grid_old[i * n + j] = 0.0;
        }
    }
    for (unsigned int i = 0; i < n; i++) {
        unsigned int top_id = i;
        unsigned int bottom_id = (n - 1) * n + i;
        unsigned int left_id = i * n;
        unsigned int right_id = i * n + (n - 1);
        grid_new[bottom_id] = grid_new[left_id] = grid_new[right_id] = 273.0;
        grid_new[top_id] = 400.0;
        grid_old[bottom_id] = grid_old[left_id] = grid_old[right_id] = 273.0;
        grid_old[top_id] = 400.0;
    }
}

void save_grid_to_file(const std::vector<float> &grid, unsigned int n, const std::string &filename) {
    std::ofstream f(filename);
    if (!f.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }
    f << std::fixed << std::setprecision(6);
    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < n; j++) {
            f << grid[i * n + j] << " ";
        }
        f << "\n";
    }
    f.close();
}

void CPUTimer::start_timer() {
    this->start = clock();
}

void CPUTimer::stop_timer() {
    this->stop = clock();
}

float CPUTimer::get_elaspsed_time_in_ms() {
    return (float)(this->stop - this->start) * 1000 / CLOCKS_PER_SEC;
}