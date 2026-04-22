# Chapter 2: Baselines and Warp-State Profiling Notes

## Scope
This report covers three tasks:
1. CPU baselines on box-scaled and dragon in sequential and parallel mode.
2. GPU baselines on dragon scene with block sizes 8x8, 16x16, and 32x32.
3. Profiler script update for Warp State divergence evidence collection.

## Automation and Reproducibility
- Baseline runner script: [chapter2_run_baselines.sh](chapter2_run_baselines.sh)
- Profiler script: [profiler_local_run.sh](profiler_local_run.sh)
- Baseline output folder used for this report: [benchmark_logs/chapter2_20260422_083443](benchmark_logs/chapter2_20260422_083443)
- Baseline CSV: [benchmark_logs/chapter2_20260422_083443/results.csv](benchmark_logs/chapter2_20260422_083443/results.csv)
- Profiler output folder used for this report: [benchmark_logs/ncu_20260421_211237](benchmark_logs/ncu_20260421_211237)
- Profiler log: [benchmark_logs/ncu_20260421_211237/ncu_run.log](benchmark_logs/ncu_20260421_211237/ncu_run.log)

## Baseline Results
| Case               | Mode           | Scene      | Block | Status    | Render Time (s) | Total Rays | Throughput (MRays/s) |
| ------------------ | -------------- | ---------- | ----- | --------- | --------------: | ---------: | -------------------: |
| cpu_seq_box_scaled | cpu-sequential | box-scaled | 16x16 | OK        |          4.0007 |  1,152,594 |                0.288 |
| cpu_seq_dragon     | cpu-sequential | dragon     | 16x16 | OK        |          4.1230 |    882,271 |                0.214 |
| cpu_par_box_scaled | cpu-parallel   | box-scaled | 16x16 | OK        |          4.0661 |  1,152,594 |                0.283 |
| cpu_par_dragon     | cpu-parallel   | dragon     | 16x16 | OK        |          4.1804 |    882,271 |                0.211 |
| gpu_dragon_8x8     | gpu            | dragon     | 8x8   | OK        |          0.0817 |    882,895 |               10.808 |
| gpu_dragon_16x16   | gpu            | dragon     | 16x16 | OK        |          0.0912 |    882,694 |                9.674 |
| gpu_dragon_32x32   | gpu            | dragon     | 32x32 | FAILED(1) |               - |          - |                    - |

## Comments on the Baselines
- CPU parallel mode did not improve throughput in this run set:
  - box-scaled: 0.288 -> 0.283 MRays/s
  - dragon: 0.214 -> 0.211 MRays/s
  - Is it just meassurement error???
- GPU is much faster than CPU for dragon:
  - 8x8 block vs CPU sequential dragon: about 50.5x higher throughput (10.808 / 0.214)
  - 16x16 block vs CPU sequential dragon: about 45.2x higher throughput (9.674 / 0.214)
- On this hardware/kernel, 8x8 outperformed 16x16 by about 11.7% for dragon.

## 32x32 GPU Failure Evidence
The 32x32 configuration failed with launch resource limits.

Extracted from [benchmark_logs/chapter2_20260422_083443/gpu_dragon_32x32.log](benchmark_logs/chapter2_20260422_083443/gpu_dragon_32x32.log):

    `numba.cuda.cudadrv.driver.CudaAPIError: [701] Call to cuLaunchKernel results in CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES`

This is a valid benchmark outcome for the tested device and kernel footprint, and it is now captured in the CSV status field instead of aborting the whole run.

## Warp-State Profiling Script Changes and Status
- TODO

