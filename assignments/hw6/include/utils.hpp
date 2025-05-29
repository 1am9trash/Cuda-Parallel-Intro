#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>
#include <ctime>


void randomize_vector(std::vector<float> &vec, float mx, float mn);
void randomize_vector_with_exp_distribution(std::vector<float> &vec, float mx);
std::vector<float> create_matrix(unsigned int m, unsigned int n, float mx, float mn);
std::vector<float> create_vector(unsigned int m, float mx, float mn);
std::vector<float> create_vector_with_exp_distribution(unsigned int m, float mx);
bool check_matrix_same(std::vector<float> &a, std::vector<float> &b);
void print_matrix(std::vector<float> mat, unsigned int limit);

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