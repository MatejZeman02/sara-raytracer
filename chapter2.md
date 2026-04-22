# Chapter 2: Baselines and Warp-State Profiling Notes

## Scope
This report covers three tasks:
1. CPU baselines on box-scaled and dragon in sequential and parallel mode.
2. GPU baselines on dragon scene with block sizes 8x8, 16x16, and 32x32.
3. Profiler script update for Warp State divergence evidence collection.

## Automation and Reproducibility
- Baseline runner script: [chapter2_run_baselines.sh](chapter2_run_baselines.sh)
- Profiler script: [profiler_local_run.sh](profiler_local_run.sh)
- Baseline output folder used for this report: [benchmark_logs/chapter2_20260422_151422](benchmark_logs/chapter2_20260422_151422)
- Baseline CSV: [benchmark_logs/chapter2_20260422_151422/results.csv](benchmark_logs/chapter2_20260422_151422/results.csv)
- Manual profiler output folder used for this report: [benchmark_logs/ncu_manual](benchmark_logs/ncu_manual)
- Nsight Compute report: [benchmark_logs/ncu_manual/profile_report.ncu-rep](benchmark_logs/ncu_manual/profile_report.ncu-rep)
- Warp-state CSV export: [benchmark_logs/ncu_manual/ncu_warpstate.csv](benchmark_logs/ncu_manual/ncu_warpstate.csv)

## Baseline Results
| Case               | Mode           | Scene      | Block | Status    | Render Time (s) | Total Rays | Throughput (MRays/s) |
| ------------------ | -------------- | ---------- | ----- | --------- | --------------: | ---------: | -------------------: |
| cpu_seq_box_scaled | cpu-sequential | box-scaled | n/a   | OK        |          3.8963 |  1,152,594 |                0.296 |
| cpu_seq_dragon     | cpu-sequential | dragon     | n/a   | OK        |         21.5750 |  2,509,144 |                0.116 |
| cpu_par_box_scaled | cpu-parallel   | box-scaled | n/a   | OK        |          2.4125 |  1,152,594 |                0.478 |
| cpu_par_dragon     | cpu-parallel   | dragon     | n/a   | OK        |          5.3934 |  2,509,144 |                0.465 |
| gpu_dragon_8x8     | gpu            | dragon     | 8x8   | OK        |          0.2960 |  2,510,034 |                8.481 |
| gpu_dragon_16x16   | gpu            | dragon     | 16x16 | OK        |          0.3293 |  2,509,867 |                7.622 |
| gpu_dragon_32x32   | gpu            | dragon     | 32x32 | FAILED(1) |               - |          - |                    - |

## Comments on the Baselines
- CPU parallel mode improves throughput:
  - box-scaled: 0.296 -> 0.478 MRays/s (1.61x)
  - dragon: 0.116 -> 0.465 MRays/s (4.01x)
- Corrected dragon run has about 2.51M rays (not about 0.88M), which is consistent with the cluster-scale dragon workload.
- On this hardware/kernel, 8x8 outperformed 16x16 by about 11.3% for dragon.

## 32x32 GPU Failure Evidence
The 32x32 configuration failed with launch resource limits.

Extracted from [benchmark_logs/chapter2_20260422_151422/gpu_dragon_32x32.log](benchmark_logs/chapter2_20260422_151422/gpu_dragon_32x32.log):

    numba.cuda.cudadrv.driver.CudaAPIError: [701] Call to cuLaunchKernel results in CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES

This is a valid benchmark outcome for the tested device and kernel footprint, and it is now captured in the CSV status field instead of aborting the whole run.

## Warp-State Profiling Results (Manual Local NCU Run)
Warp-state metrics were successfully collected from [benchmark_logs/ncu_manual/ncu_warpstate.csv](benchmark_logs/ncu_manual/ncu_warpstate.csv).

The CSV contains two kernel IDs:
- ID 0: warmup launch (`(1,1,1)` block/grid), not representative.
- ID 1: measured render kernel launch (`Block=(16,16,1)`, `Grid=(50,50,1)`), used for analysis below.

### Processed Warp-State Metrics (ID 1)
| Metric                                   |             Value |
| ---------------------------------------- | ----------------: |
| Warp Cycles Per Issued Instruction       |       7.42 cycles |
| Warp Cycles Per Executed Instruction     |       7.44 cycles |
| Avg. Active Threads Per Warp             | 8.38 / 32 (26.2%) |
| Avg. Not Predicated Off Threads Per Warp | 7.93 / 32 (24.8%) |

### Main Findings from Nsight Rules
- Dominant stall reason: scoreboard dependency on L1TEX operations, estimated at 42.16% of total issue-cycle stalls.
- Thread divergence / predication remains significant: only about 8 active threads per warp on average, with Nsight flagging a divergence optimization opportunity (rule estimate: 33.02).

### Interpretation for the Renderer
- Memory-side latency (L1TEX scoreboard waits) is currently the primary throughput limiter for this kernel launch.
- Warp execution efficiency is low (~25% of lanes active), which matches divergent control flow expected in path tracing.
- Next optimization direction should prioritize:
  - reducing divergent paths inside a warp,
  - improving data locality/coalescing for traversal/material fetches,
  - considering wavefront/path compaction in later architecture iterations.

