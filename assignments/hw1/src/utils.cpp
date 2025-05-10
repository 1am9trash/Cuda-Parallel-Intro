#include <vector>
#include <random>
#include <iostream>
#include <ctime>
#include <algorithm>
#include "utils.hpp"


#define RAND_SEED 1337

void randomize_matrix(std::vector<float> &mat, float mx, float mn) {
    static std::default_random_engine rng(RAND_SEED);
    std::uniform_real_distribution<float> dist(mn, mx);
    std::generate(mat.begin(), mat.end(), [&]{ return dist(rng); });
}

std::vector<float> create_matrix(unsigned int m, unsigned int n, float mx, float mn) {
    std::vector<float> mat(m * n);
    randomize_matrix(mat, mx, mn);
    return mat;
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

void CPUTimer::start_timer() {
    this->start = clock();
}

void CPUTimer::stop_timer() {
    this->stop = clock();
}

float CPUTimer::get_elaspsed_time_in_ms() {
    return (float)(this->stop - this->start) * 1000 / CLOCKS_PER_SEC;
}