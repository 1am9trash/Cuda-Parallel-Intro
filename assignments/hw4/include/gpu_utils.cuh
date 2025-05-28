#ifndef GPU_UTILS_CUH
#define GPU_UTILS_CUH

#include <cuda_runtime.h>
#include "utils.hpp"

class GPUTimer: public ITimer {
private:
    cudaEvent_t start;
    cudaEvent_t stop;
public:
    GPUTimer();
    ~GPUTimer();
    void start_timer() override;
    void stop_timer() override;
    float get_elaspsed_time_in_ms() override;
};

#endif