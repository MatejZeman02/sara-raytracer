# NI-PG1 Matej Zeman (zemanm40) 2026

My Sara raytracer in numba-cuda.

## Structure

Separate homeworks are in the folders, merged from branch with the same name.
Use `run_local.sh hw0x` script to run specific homework (on linux). I managed to run it on FIT cluster as well.

The output will be in the homework folder as `output.ppm/png`.

## Render Times

Testing on my box-advanced with +-5000 triangles.
(cpu side is usually constant)

Dimensions per machine:
- 1440**2 on my local rtx3070 mobile.
- 2880**2 on A100 on cluster.


## Homework render times:
Pure rendering on both machines:
| hw | local | remote |
| - | - | - |
|1 | 2.80 s| 1.50 s |
|2 | xx s| xx s |
|3 | xx s| xx s |
|4 | xx s| xx s |
|5 | xx s| xx s |

