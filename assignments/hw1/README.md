HW 1 – Matrix Reciprocal Addition
---

## Assignment Brief
Implement and benchmark a **modified matrix-addition**.

$$
c_{ij} = \frac{1}{a_{ij}} + \frac{1}{b_{ij}}, \quad a, b, c\in\mathbb{R}^{N\times N}, \quad N = 6400
$$

## Methodology
I wrote two CUDA kernels to solve the problem. They differ only in how much work each thread does and, therefore, in the way they touch memory. 

### Scalar Kernel
- Each thread **loads 1 float** from `A`, **loads 1 float** from `B`, calculates `1/a + 1/b`, and **stores 1 float** to `C`.  
- A single load is only 4 bytes, while the DRAM transaction size is 128 bytes; however, every warp (32 threads) accesses **32 consecutive floats**. The hardware therefore **coalesces all 32 scalar loads into one 128-byte transaction**, so no extra memory traffic is generated.

```c
__global__ void add_matrix_kernel(
    const float *a,
    const float *b,
    float *c,
    unsigned int n
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n * n) {
        c[index] = 1 / a[index] + 1 / b[index];
    }
}
```

### Vectorized Kernel
- Each thread **loads 4 floats** from `A` and `B` using a `float4` vector read, computes four results, and **stores 4 floats** to `C`.
- This packing removes **~75 %** of the **address-calculation instructions** and gives higher **instruction-level parallelism (ILP)**.
- In this implementation the base address must be **16-byte aligned** and the total number of elements must be a **multiple of 4** (satisfied when `N = 6400`).

```c
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>((float *)pointer)[0])

__global__ void vectorized_add_matrix_kernel(
    const float *a,
    const float *b,
    float *c,
    unsigned int n
) {
    const unsigned int index = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (index < n * n) {
        float4 reg_a = FETCH_FLOAT4(&a[index]);
        float4 reg_b = FETCH_FLOAT4(&b[index]);
        float4 reg_c = FETCH_FLOAT4(&c[index]);
        reg_c.x = 1 / reg_a.x + 1 / reg_b.x;
        reg_c.y = 1 / reg_a.y + 1 / reg_b.y;
        reg_c.z = 1 / reg_a.z + 1 / reg_b.z;
        reg_c.w = 1 / reg_a.w + 1 / reg_b.w;
        FETCH_FLOAT4(&c[index]) = reg_c;
    }
}
```

## Experiment Results

### CPU vs. GPU (Best Case)

The first table highlights the headline speed-up a GPU offers over a CPU when **all overheads are included**. It answers the practical question: "How much wall-clock time do I save by moving this workload to the GPU?"

| Platform         | Configuration    | Total Time (ms) | Speed-up        |
|------------------|------------------|-----------------|-----------------|
| CPU              | –                | 148.174         | 1.0× (baseline) |
| GPU – Scalar     | `blockDim = 256` | 82.037          | 1.8×            |
| GPU – Vectorized | `blockDim = 128` | 81.957          | 1.81×           |

> Note GPU totals include host↔device transfers; pure GPU compute is only ~3 ms.

### GPU timing details

The second table breaks down GPU time for four block sizes and the two kernels. It shows whether kernel design or launch parameters have any measurable influence.

| Block size (threads / block) | Kernel     | H2D IO (ms) | Compute (ms) | D2H IO (ms) | Total Time (ms) |
|------------------------------|------------|-------------|--------------|-------------|-----------------|
| **128**                      | Scalar     | 50.066      | 3.241        | 28.754      | 82.061          |
|                              | Vectorized | 50.067      | 3.146        | 28.745      | 81.957          |
| **256**                      | Scalar     | 50.054      | 3.236        | 28.748      | 82.037          |
|                              | Vectorized | 50.072      | 3.144        | 28.757      | 81.973          |
| **512**                      | Scalar     | 50.082      | 3.249        | 28.800      | 82.131          |
|                              | Vectorized | 50.058      | 3.146        | 28.769      | 81.974          |
| **1024**                     | Scalar     | 50.081      | 3.287        | 28.790      | 82.158          |
|                              | Vectorized | 50.078      | 3.147        | 28.747      | 81.973          |


## Findings & Discussion  

### GPU vs. CPU 
- **Speed-up:** 148 ms → 82 ms (~1.8 ×).  
- **Take-away:** 95 % of the GPU time is I/O or allocation. Once PCIe traffic is saturated, extra SMs or fancier kernels change almost nothing.

### Scalar vs. Vectorized
- **Same bytes on the wire**
  - Both kernels move 4 bytes per matrix element.  
  - Vector loads (`float4`) simply bundle four neighbouring floats that the hardware would already fetch together.  
- **Instruction count drops, but it is not the bottleneck**  
  - Vectorization removes ~75 % of address-calculation instructions. 
  - DRAM latency still dominates, so the ~3 % compute gain is buried under I/O time.  
- **When would vectorization help more?**  
  - **Shared-memory reuse**: For example, a block-GEMM kernel loads a tile into shared memory and reuses it many times. Once global-memory traffic is amortised, the bottleneck becomes instruction scheduling and register moves; cutting 75 % of address-calculation instructions with vector loads can then deliver a clear speed-up.

### Block size sweep – negligible effect  
- Tested `blockDim.x` = 128, 256, 512, 1024.  
- Total runtime changed by **<0.2 %**.  
- Registers stayed within budget; DRAM bandwidth was already saturated.  

These observations confirm that for **bandwidth-limited**, single-pass array operations the key to speed is **moving less data**, not **rewriting the kernel**.
