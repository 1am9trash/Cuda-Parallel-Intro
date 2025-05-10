#include <cuda_runtime.h>
#include "gpu_utils.cuh"


GPUTimer::GPUTimer() {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
}

GPUTimer::~GPUTimer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void GPUTimer::start_timer() {
    cudaEventRecord(start, 0);
}

void GPUTimer::stop_timer() {
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
}

float GPUTimer::get_elaspsed_time_in_ms() {
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}