# Numba Raytracer
## Chapter 1

### 1.1 Problem Definition: Ray tracer
The objective of my project is implement a cpu/Nvidia gpu ray-tracing engine in numba, capable of rendering complex 3D geometry such as a million-triangles scenes. Ray tracing is inherently highly parallel problem, as the light transport for each pixel can theoretically be calculated independently.

TODO: explain briefly steps of generic raytracer. From each pixel of desired resulting image, we shoot a 3D ray that collides with triangles in the scene or goes into infinity (primary ray). Based on the material of the triangle it hits we can reflect/refract the primary ray, creating secondary rays. When the ray stops bouncing, we calculate shading by creating another (shadow) rays from the final 3D spot that goes to the light emmiting pixels. Asking how much light will the spot absorb. These are called samples, because we are sampling the lights area uniformly using Monte Carlo approximation of the integral. The final shading consists of diffuse and specular elements. The ambient or global illumination is not used in ray tracer, but in rasterized workflow or in proper path tracer. Meaning if pixel is occluded and the light doesn't reach it, it will be pure black, even if you put reflective material nearby, that would normally add some bounce light to the dark spot.

However, the technical challenge of implementing ray tracer lies in the high variance of the computational workload. Light rays such as deep glass refractions or trapped bounces inside detailed geometry require drastically more Bounding Volume Hierarchy (BVH) incidence tests than standard sky pixels that won't interact with any triangles at all. This imbalance, of course, causes thread/warp divergence.

### 1.2 Description of the sequential algorithm and implementation
The foundational algorithm is a physically based ray tracer. For every pixel in the image frame, a primary ray is generated and cast into the scene, traversing a static BVH to efficiently find triangle intersections. Upon intersection, the ray scatters according to the surface material properties, and the process repeats iteratively until it hits a light source or exceeds a maximum bounce limit or light attenuation doesn't add any more color.

**Implementation via Numba** The engine is implemented in Python, specifically utilizing the *Numba Just-In-Time (JIT)* compiler. By strictly avoiding dynamic memory allocations and packing all scene data (geometry, materials, and BVH nodes) into flat, C style contiguous NumPy arrays, the core tracing loop was designed to be highly parallel-friendly from the ground up.

This memory-safe architecture allows the exact same intersection and shading math to scale seamlessly. A purely sequential approach can be easily converted to utilize multi-core CPUs via the `@njit(parallel=True)` decorator. By replacing the decorator with `@cuda.jit` and changing pure numpy arrays to cuda allocations, the sequential loop transitions into a parallel single (mega) kernel without requiring a total rewrite of underlying math.

**Disadvantages of Python compared to C++** While Numba provides near-native execution speeds, choosing Python over a systems language like C++ introduces notable disadvantages in high-performance rendering:
- **Memory Control:** C++ offers explicit control over pointers, cache-line alignment, and hardware registers, which are critical for optimizing BVH traversal. Numba abstracts these low-level details away, making memory tuning difficult.
- **Data Structure Limitations:** Building and manipulating recursive, pointer-heavy tree structures directly on the device is highly restrictive in Numba CUDA compared to native C++ CUDA, forcing heavy reliance on host-side preprocessing.
- **Host-Side Overhead:** Python's Global Interpreter Lock and general runtime overhead complicate data preparation and stream compaction phases, creating CPU bottlenecks that would be trivial to multi-thread natively in C++.
- **Object-Oriented Programming:** Inside the kernel no inheritance or custom class was used. They are currently experimental in numba and cause severe performance degradation. So the code is structured in a more C-style, with functions and flat arrays instead of objects and pointers. It not so Pythonic way to call long functions with a ton of parameters and even casting floats to 32 bits, but numba simply doesn't yet implement better ways to keep the performance under the control.

***

## Chapter 2: GPU Implementation and architectural Optimization

### 2.1 Transition to GPU and Baseline Benchmarks
The testing scenes are based on the famous Cornell box with some other objects inside. My local notebook Nvidia card is RTX 3070 mobile and processor 11th Gen Intel® Core™ i7-11800H × 16 which is sufficent for comparing the cpu/gpu implementation. By structuring the scene data as flat, contiguous arrays, the core ray-tracing loop was launched as a massively parallel "Megakernel." Initial benchmarks on the Dragon scene demonstrated quite big performance uplift compared to the CPU baselines.

| Mode           | Scene      | cores/Block | Render Time (s) | Total Rays | Throughput (MRays/s) |
| :------------- | :--------- | :---------- | --------------: | ---------: | -------------------: |
| cpu-sequential | box-scaled | 1           |          3.8963 |  1,152,594 |                0.296 |
| cpu-sequential | dragon     | 1           |         21.5750 |  2,509,144 |                0.116 |
| cpu-parallel   | box-scaled | 16          |          2.4125 |  1,152,594 |                0.478 |
| cpu-parallel   | dragon     | 16          |          5.3934 |  2,509,144 |                0.465 |
| gpu            | dragon     | 8x8         |          0.2960 |  2,510,034 |                8.481 |
| gpu            | dragon     | 16x16       |          0.3293 |  2,509,867 |                7.622 |
| gpu            | dragon     | 32x32       |               - |          - |                    - |

**Baselines:**
- CPU parallel mode successfully improves throughput across scenes (up to 4.01x on the Dragon scene).
- The GPU Megakernel vastly outperforms the CPU implementation. On this specific hardware/kernel combination, the 8x8 block configuration outperformed 16x16 by roughly 11%.
- **The 32x32 Failure:** The 32x32 configuration failed with `CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES`. This is a valid benchmark outcome demonstrating a hard launch resource limit. The Megakernel simply requires too many physical hardware registers per thread to launch 1,024 threads simultaneously in a single block.

## 2.2 Profiling the Divergence Bottleneck
Despite the speedup over the CPU, baseline profiling using Nvidia Nsight Compute (NCU) revealed severe architectural inefficiency within the Megakernel.

| Metric                                   |                 Value |
| :--------------------------------------- | --------------------: |
| Warp Cycles Per Issued Instruction       |           7.42 cycles |
| Warp Cycles Per Executed Instruction     |           7.44 cycles |
| **Avg. Active Threads Per Warp**         | **8.38 / 32 (26.2%)** |
| Avg. Not Predicated Off Threads Per Warp |     7.93 / 32 (24.8%) |

It is important to note, that we don't actually know how does ncu meassure these statistics and then we do not know how reliable they are. But, as the metrics indicate, the Megakernel averaged only **8.38 active threads per warp (~26.2%)**. The thread divergence occurred because the megakernel is not optimized for GPU warps. while most rays escaped to the sky or hit simple geometry, a minority of complex rays became trapped in deep glass refractions or dense BVH traversals, forcing the rest of the warp to sit idle. Furthermore, NCU pointed 'L1TEX scoreboard dependency waits' as the primary stall reason, resulting from the chaotic memory reads of BVH. Unfortunatelly, connecting to the gpu process require sudo righs, so I was not able to meassure it on the schoold cluster with the proper graphical cards like A100 and RTX 4070 Ti.

### 2.3 Wavefront Stream Compaction & The U-Curve
To mitigate this divergence, the algorithm was refactored from a Megakernel into a two-pass **Wavefront (Stream Compaction)** architecture:

1. **Pass 1:** Enforces a strict BVH operations **budget**. Easy rays finish and terminate. Complex rays pause, spilling their exact state (ray origin, direction, accumulated color) to global device memory.
2. **CPU Compaction:** The Python host extracts the active ray indices using NumPy to create a dense array of unfinished work. It could be another kernel, but the cpu overhead with running another kernel even if Cupy would be used would be slower than just sorting unfinished on cpu with numpy. Two sorting algorithms were implemented: one that compacts rays hiting the same material and second, ordered by ray direction. Ray direction showed up to be slightly worse in all cases, so it doesn't appear it any of these metrics.
3. **Pass 2 (Cleanup):** Relaunches exclusively over the compacted array, completing the deep bounces. If we were to run the ncu command for thread divergence on the second kernel, it could be even worse than for the first one, because all these remaining threads are non-trivial. But if we exclude the CPU work and kernel launch overhead (which in python is much larger), then it should be more efficient.

An automated parameter sweep across the operations budget revealed a tradeoff between compute efficiency (resolving short rays early) and VRAM bandwidth overhead. 

| Case                    | Ops Budget | Total Render Time (s) | Pass 1 Time | Compaction | Pass 2 Time | Active Rays Compacted |
| :---------------------- | :--------- | :-------------------- | :---------- | :--------- | :---------- | :-------------------- |
| **Megakernel Baseline** | n/a        | 0.152                 | -           | -          | -           | -                     |
| Wavefront               | 100        | 0.4326                | 0.0080s     | 0.0014s    | 0.4233s     | 573,774               |
TODO: fill the the table with the results on cluster

![4070_0.png](4070_0.png)
*[wavefront vs megakernel render time (nvidia_geforce_rtx_4070_ti)]*

TODO: fact check:
As seen in the Ada Lovelace (RTX 4070 Ti) performance graph, increasing the budget shifts the workload from Pass 2 back to Pass 1. Around a budget of `4000`, the total render time drops below the Megakernel baseline (0.152s) and enters the target zone. A budget of `8000` proved highly optimal for true Stream Compaction, successfully resolving ~97% of standard rays in Pass 1 and handing off only ~86,000 deep rays to Pass 2.

### 2.4 The "Chaos Kernel" Phenomenon
Despite the overall render time decreasing on newer hardware, Nsight Compute profiling on Pass 2 revealed an unexpected case: warp efficiency actually *worsened*. Active threads dropped from the baseline 8.38 down to **4.68 per warp (~14.6%)**.

This was diagnosed as the **"Chaos Kernel"** phenomenon (link?). While stream compaction successfully grouped *active* rays together, it completely destroyed memory access patterns. Adjacent threads in the newly compacted array now contained rays pointing in entirely different directions, hitting opposite sides of the BVH. This maximized memory thrashing in the L1/L2 cache and exacerbated spatial divergence, proving that packing threads without proper sorting introduces severe secondary bottlenecks.

### 2.5 Data Locality: CPU Sorting and "The Shredder Effect"
To resolve spatial divergence, a material and directional sorting phase was introduced between Pass 1 and Pass 2. 

Because the compacted array was relatively small (~86k elements), a "Skinny Round-Trip" strategy was utilized: transferring the indices to the CPU, running a dependency-free `numpy.argsort`, and transferring them back. This operation cost only 3–9 milliseconds, proving highly viable compared to the complexity of writing a native CUDA radix sort.

However, implementing the sort revealed another architectural trap: **The Shredder Effect**. Launching Pass 2 with the standard 2D block geometry (e.g., 16x16) forced the CUDA scheduler to map a 2D thread grid onto the newly sorted 1D array. This mapping effectively "shredded" the contiguous memory layout, nullifying the sort. To ensure the hardware respects the software sorting order, Pass 2 must utilize a strictly 1D execution model (`cuda.grid(1)`) with 1D block sizes.

### 2.6 Cross-Generational Hardware Anomalies (The Ampere Collapse)
During validation across different devices, a massive hardware-level anomaly emerged. The Wavefront architecture was benchmarked against the Megakernel on three different GPUs: an RTX 3070 Mobile (Ampere), an A100 40GB (Ampere), and an RTX 4070 Ti (Ada Lovelace).

![3070.png](3070.png)
*[wavefront vs megakernel render time (nvidia_geforce_rtx_3070_laptop_gpu)]* 

![A100.png](A100.png)
*[wavefront vs megakernel render time (nvidia_a100-pcie-40gb)]*

As shown in the Ampere graphs above, the total render time (black line) *never* drops below the Megakernel baseline (red dashed line). Even on the enterprise-grade A100, the baseline of `0.229s` was entirely unbeatable by the Wavefront approach. Conversely, on the RTX 4070 Ti, the bounded Pass 1 kernel effortlessly shattered the Megakernel baseline, reaching speeds near `0.11s`.

To determine if the NVVM compiler was aggressively unrolling loops for newer architectures, the raw PTX assembly was extracted for both kernels.

**PTX Assembly Extraction Analysis**

| Metric                          | Megakernel | Wavefront Pass 1 |
| :------------------------------ | :--------- | :--------------- |
| **Line Count**                  | 2,163      | 2,472            |
| **Branch Instructions (`bra`)** | 91         | 98               |
| **32-bit Registers (`%r`)**     | 56         | 132              |
| **64-bit Registers (`%rd`)**    | 527        | 825              |
| **Predicate Registers (`%p`)**  | 160        | 182              |
| **Float Registers (`%f`)**      | 1,244      | 1,333            |

**Architectural Conclusions:**
1. The compiler generated almost identical virtual assembly for both architectures, disproving some hypothetic optimization.
2. The bounded kernel induced massive register pressure, jumping from 527 to 825 64-bit registers per thread.
3. The discrepancy probably lies in the PTX-to-SASS binary translation and the physical cache hierarchy. Ampere hardware (A100 and 3070) suffers from fatal register spilling to local memory under this 825-register footprint. The VRAM read/write overhead completely erases any compute cycles saved by Stream Compaction. The RTX 4070 Ti (Ada Lovelace), featuring a redesigned Streaming Multiprocessor and a massively expanded L2 cache, effortlessly absorbs the register pressure, resulting in superior execution speeds for instruction-heavy ray-tracing kernels. Although it could be anything else really.
