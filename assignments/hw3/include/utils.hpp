#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>
#include <ctime>
#include <string>


void randomize_vector(std::vector<float> &vec, float mx, float mn);
std::vector<float> create_matrix(unsigned int m, unsigned int n, float mx, float mn);
std::vector<float> create_vector(unsigned int m, float mx, float mn);
bool check_matrix_same(std::vector<float> &a, std::vector<float> &b);
void print_matrix(std::vector<float> mat, unsigned int limit);

void initialize_grid(double *grid_new, double *grid_old, double *grid_rho, const unsigned int n, const double h);
void save_grid_to_file(const std::vector<double> &grid, unsigned int n, const std::string &filename);

class ITimer {
public:
    virtual void start_timer() = 0;
    virtual void stop_timer() = 0;
    virtual float get_elaspsed_time_in_ms() = 0;
};

class CPUTimer: public ITimer {
private:
    std::clock_t start;
    std::clock_t stop;
public:
    void start_timer() override;
    void stop_timer() override;
    float get_elaspsed_time_in_ms() override;
};

#endif