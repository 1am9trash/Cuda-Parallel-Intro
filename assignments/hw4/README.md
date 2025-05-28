HW 4 – Dot Product with Multi-GPU Parallelism
---

## Assignment Brief
Implement and benchmark a program to compute the **dot product** of **two real-valued vectors** using multiple GPUs and CPU.

The dot product of two vectors `a` and `b` is defined as:

$$
a = \sum_{i=0}^{N-1} a_{i}b_{i}, \quad a, b \in \mathbb{R}^{N}, \quad N = 4096000
$$

## Analysis
- Extending the approach used in HW2, we again focus primarily on **kernel execution time**, rather than total runtime which includes host–device data transfers (`cudaMemcpy`). While data movement overhead still exists, isolating kernel-level performance enables a more precise evaluation of GPU execution efficiency under different **launch configurations**.
- To assess performance scaling, I systematically vary:
  - **Number of GPUs**: To observe parallel scaling and inter-device partitioning.
  - **Threads per block**
  - **Blocks per grid**

### Why Vectorized Access Is Not Considered
1. **No data reuse**: The dot product involves no reuse of data. Each element of `a[i]` and `b[i]` is read exactly once. All memory access occurs **directly from global memory**.
2. **Low arithmetic intensity**: The computation performs only two FLOPs (multiply and add) per pair of 4-byte inputs, making it **memory bandwidth-bound**, not **compute-bound**.
3. **Limited benefit from vectorized access**: While vectorized access (e.g., `float4`) can reduce the number of load instructions, it does not alleviate the bandwidth bottleneck in this workload. Since memory access is already coalesced, the gains from vectorization are minimal.

Not implemented in this study: Given the above, vectorized access was not implemented in the experiment, as it would add complexity without meaningful performance improvement.

## Methodology
This experiment uses **OpenMP** with `cudaSetDevice()` to enable parallel execution across multiple GPUs. The process is as follows:
1. **Thread assignment**
   - **OpenMP** is used to launch one CPU thread per GPU, with `omp_set_num_threads(n_gpu)`. 
   - Each thread calls `cudaSetDevice(thread_id)` to bind its CUDA operations to a specific device.
2. **Data partitioning**
   - The input vectors are evenly divided among GPUs.
   - Each device allocates memory (`cudaMalloc`) and receives its data chunk via `cudaMemcpy`.
3. **Kernel execution**
   - Each GPU launches the kernel with a **fixed block size** and a portion of the total grid (`grid_size / n_gpu`).
   - The kernel computes the partial dot product independently using a grid-stride loop.
4. **Result aggregation**
   - Partial results from each GPU are copied back to the host.
   - A final reduction is performed on the CPU to compute the full dot product result.

## Experiment Results

### CPU vs. GPU (Best Case)

We compare the total wall-clock time of the CPU implementation and the full GPU pipeline, which includes:
- Host-to-device memory transfer
- Kernel execution
- Device-to-host memory transfer
- Final reduction on host

| Platform             | Total Time (ms) | Speed-up         |
|----------------------|-----------------|------------------|
| CPU                  | 3.834233        | 1.00× (baseline) |
| 1 GPU (total)        | 6.235415        | 0.61×            |
| 1 GPU (data IO)      | 6.005874        | -                |
| 1 GPU (kernel only)  | 0.226543        | -                |
| 2 GPU (total)        | 4.392420        | 0.87×            |
| 2 GPU (data IO)      | 4.270264        | -                |
| 2 GPU (kernel only)  | 0.118458        | -                |

**Kernel Performance**
- With 2 GPUs, kernel time is nearly halved compared to the single-GPU case, indicating ideal parallel scaling and efficient workload partitioning.

**Data Transfer Time**
1. Data transfer does not scale linearly. While kernel time benefits significantly from GPU scaling, total data transfer time improves by only **~29%** when moving from 1 to 2 GPUs.
2. This suggests potential **bandwidth contention**, likely caused by system-level constraints such as a shared PCIe root complex or memory controller.
3. As I lacked permission to inspect the **system topology** directly, I conducted an experiment to verify this hypothesis:
   - Transfer 2 GB per GPU **sequentially** (1 GPU at a time).
   - Transfer 2 GB per GPU **concurrently** using two threads.

   The results below confirm degraded bandwidth during concurrent transfers:

   |              | Total Time (ms) | Bandwidth (GB/s) |
   |--------------|-----------------|------------------|
   | Single GPU 0 | 248.43 ms       | 8.05             |
   | Single GPU 1 | 248.37 ms       | 8.05             |
   | Dual GPU 1   | 291.67 ms       | 6.86             |
   | Dual GPU 1   | 318.18 ms       | 6.29             |

### GPU Kernel

> `GPU = 1`

| Block Size | Grid Size | Compute Time (ms) |
|------------|-----------|-------------------|
| 32         |   128000  | 0.558247          |
|            |    64000  | 0.313941          |
|            |    32000  | 0.257079          |
| 64         |    64000  | 0.329411          |
|            |    32000  | 0.231588          |
|            |    16000  | 0.221607          |
| 128        |    32000  | 0.346762          |
|            |    16000  | 0.232250          |
|            |     8000  | 0.220972          |
| 256        |    16000  | 0.388357          |
|            |     8000  | 0.243233          |
|            |     4000  | 0.222313          |
| 512        |     8000  | 0.447366          |
|            |     4000  | 0.270179          |
|            |     2000  | 0.226543          |
| 1024       |     4000  | 0.551840          |
|            |     2000  | 0.329098          |
|            |     1000  | 0.256633          |

> `GPU = 2`

| Block Size | Grid Size | Compute Time (ms) |
|------------|-----------|-------------------|
| 32         | 128000    | 0.286925          |
|            |  64000    | 0.176080          |
|            |  32000    | 0.133842          |
| 64         |  64000    | 0.167197          |
|            |  32000    | 0.117997          |
|            |  16000    | 0.112470          |
| 128        |  32000    | 0.175435          |
|            |  16000    | 0.118948          |
|            |   8000    | 0.113715          |
| 256        |  16000    | 0.196001          |
|            |   8000    | 0.125036          |
|            |   4000    | 0.114912          |
| 512        |   8000    | 0.225734          |
|            |   4000    | 0.139113          |
|            |   2000    | 0.118458          |
| 1024       |   4000    | 0.278445          |
|            |   2000    | 0.169146          |
|            |   1000    | 0.132679          |

**Optimal Block Size: 64–128**
- This range consistently delivers the best performance, balancing thread-level parallelism and resource utilization.

**Smaller Grid Size Performs Better**
- Assigning more work per thread (i.e., fewer threads overall) leads to better performance by reducing overhead.
