HW 2 – Matrix Trace via Parallel Reduction 
---

## Assignment Brief
Implement and benchmark a program to compute the **trace of a matrix** using both CPU and GPU.

The trace of a matrix `a` is defined as the sum of its main diagonal elements.

$$
\text{trace}(a) = \sum_{i=0}^{N-1} a_{ii}, \quad a \in \mathbb{R}^{N \times N}, \quad N = 6400
$$

## Analysis
- As in the previous assignment, the majority of GPU runtime is spent on **data movement between host and device** (i.e., `cudaMemcpy` operations).
- Although this overhead dominates the total execution time, it is largely **unavoidable** in this setting. Therefore, our performance discussion will focus primarily on **kernel execution time**, even though it represents only a small fraction of the total time.

### Computational Intensity
On the GPU, the kernel reads the `N` diagonal elements directly from global memory and performs a reduction (summing them up). This involves:
- **N loads** from global memory (one `float` per element, 4 bytes each)
- **N – 1 additions** to compute the total sum

Thus, the arithmetic intensity is:

$$
\text{Arithmetic Intensity} = \frac{N \text{ FLOPs}}{N \times 4 \text{ bytes}} = \frac{1}{4} \text{ FLOPs/byte}
$$

This is a **low-intensity workload**, meaning each byte loaded does very little computation. As a result, the kernel is clearly **global memory-bound**.

### Data Locality
This computation also exhibits **poor spatial and temporal data locality**:
- Only one element per row (i.e., `a[i][i]`) is accessed, skipping over the rest of the matrix.
- Each accessed value is used **exactly once**, meaning no reuse (temporal locality) and no dense block access (spatial locality).

### Summary
In my view, this workload is not particularly well-suited for GPU acceleration.
- The arithmetic intensity is low, so there's limited compute per byte of data.
- Memory access is sparse (diagonal-only), which makes **coalescing** difficult without preprocessing.
- Since each value is used only once, **data reuse** and **shared memory** offer little advantage.
- Techniques like **vectorized access** or **warp-level reuse** are hard to apply.

## Methodology
The GPU kernel uses **parallel reduction** to sum the diagonal elements.
- Each thread starts from its assigned index `blockIdx.x * blockDim.x + threadIdx.x` and accumulates all diagonal elements assigned to it via **grid-stride loop**.
- Each thread stores its local sum into a shared memory array `cache[]`.
- Then, an **intra-block reduction** is performed in shared memory using pairwise summation with `__syncthreads()` between steps.

The remaining reduction of the partial block sums (in `sum[]`) is performed on the host (CPU). The number of blocks is relatively small.

```c
__global__ void matrix_trace_reduce_kernel(
    const float *a,
    float *sum,
    unsigned int n
) {
    extern __shared__ float cache[];
    unsigned int cache_id = threadIdx.x;
    unsigned int cur_id = blockIdx.x * blockDim.x + threadIdx.x;

    float tmp = 0.0;
    while (cur_id < n) {
        tmp += a[cur_id * n + cur_id];
        cur_id += blockDim.x * gridDim.x; 
    }
    cache[cache_id] = tmp;
    __syncthreads();

    unsigned int reduce_id = blockDim.x >> 1;;
    while (reduce_id != 0) {
        if(cache_id < reduce_id)
            cache[cache_id] += cache[cache_id + reduce_id]; 
        __syncthreads();
        reduce_id >>= 1;
    }
    
    if (cache_id == 0)
        sum[blockIdx.x] = cache[0];
}
```

## Experiment Results

### CPU vs. GPU (Best Case)

We compare the total wall-clock time of the CPU implementation and the full GPU pipeline, which includes:
- Host-to-device memory transfer
- Kernel execution
- Device-to-host memory transfer
- Final reduction on host

| Platform             | Total Time (ms) | Speed-up         |
|----------------------|-----------------|------------------|
| CPU                  | 0.009100        | 1.00× (baseline) |
| GPU (total)          | 27.028635       | 0.0003×          |
| GPU (kernel only)    | 0.025682        | 0.35×            |

In this case, the CPU outperforms the GPU by a large margin.  
Even if we exclude memory transfer and focus only on the kernel execution time, the GPU is still slower than the CPU.
- **Lower core frequency**: GPU Cuda cores run much slower than CPU cores. Even with fewer reduction steps (log N vs. N), each operation takes longer.
- **Launch and sync overhead**: Kernel launch and thread synchronization (`__syncthreads()`) introduce fixed costs that dominate at small scales.
- **Insufficient parallelism**: With only 6400 elements, there may not be enough active threads to fully hide latency.

### GPU Kernel
We conducted two sets of experiments to investigate how different kernel launch configurations affect performance.  

**Effect of Block Size**
- In this experiment, we fixed the total number of threads (i.e., `block size × grid size` remains constant) and varied the block size.
- When `N = 6400`, the differences in execution time were minimal and often within the margin of measurement noise. To better highlight any performance impact, we repeated the test at a larger scale (`N = 25600`), where the performance difference became slightly more noticeable.
- The results show that `block size = 256` consistently provides the best performance.

> `N = 6400`

| Block Size | Compute Time (ms) |
|------------|-------------------|
| 32         | 0.024857          |
| 64         | 0.025385          |
| 128        | 0.025068          |
| 256        | 0.025057          |
| 512        | 0.025636          |
| 1024       | 0.025843          |

> `N = 25600`

| Block Size | Compute Time (ms) |
|------------|-------------------|
| 32         | 0.044767          |
| 64         | 0.043755          |
| 128        | 0.043387          |
| 256        | 0.043001          |
| 512        | 0.043283          |
| 1024       | 0.045523          |

**Effect of Grid Size (Work per Thread)**
- We also tested different grid sizes by changing how much work each thread performs. For example, a thread may process 1, 2, 4, or more diagonal elements in a loop.
- The results show **no significant performance difference** across configurations. 
