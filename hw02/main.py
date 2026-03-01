"""Homework renderer entrypoint."""

# pylint: disable=too-many-locals, unused-variable, line-too-long

# system lib imports:
import time

# global start time for total execution timing
LOAD_PYTHON_TIME = 1.0  # adjust for python load time in timing output
t_start = time.perf_counter() - LOAD_PYTHON_TIME

import os
import sys
import warnings
import math

# 3rd party imports:
import numpy as np
from PIL import Image
from numba import cuda
from numba.core.errors import NumbaPerformanceWarning

# ignore 'low occupancy' warning during the 1x1 warmup run
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

# project root in sys.path so 'utils' can be found
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# local imports:
from camera import build_camera_vectors
from render_kernel import render_kernel
from settings import BLOCK_THREADS, DIMENSION, RENDER_NON_BVH_STATS
from bvh import build_bvh
from utils.obj_loader import load_scene
from utils.ppm import save_ppm
from utils.kernel_manager import KernelManager


def _phase_time(label: str, t0: float) -> float:
    """Print elapsed time from t0 and return new timestamp."""
    t1 = time.perf_counter()
    print(f"[timing] {label:<20}: {t1 - t0:7.2f} s")
    return t1


def main():
    """
    Main function that orchestrates the ray tracing rendering pipeline.

    Function loads a 3D scene from a JSON configuration file, transfers geometry
    and material data to GPU memory, sets up camera parameters, and executes a CUDA
    kernel to render the scene. The resulting framebuffer is then saved as both PPM
    and PNG image files.

    Steps:
    - Initializes render dimensions based on DIMENSION constant
    - Loads scene data (triangles, materials, camera configuration) from JSON
    - Transfers all geometry and material data to GPU device memory (d_* variables)
    - Builds camera coordinate system vectors from camera data
    - Configures CUDA thread blocks and grid dimensions for parallel execution
    - Translate run to ensure JIT compilation is done before timing
    - Executes the render_kernel on GPU to perform ray tracing
    - Copies rendered framebuffer from GPU back to host memory
    - Saves the output as both PPM and PNG formats
    - Reports render execution time
    """
    width, height = int(DIMENSION), int(DIMENSION)

    # load scene data
    # json_file = os.path.join(project_root, "box-advanced", "setup.json")
    json_file = os.path.join(project_root, "box-sphere-original", "setup.json")
    triangles, tri_normals, mat_indices, materials, cam_data = load_scene(json_file)
    origin, p00, qw, qh, light_pos, light_color = build_camera_vectors(
        cam_data, width, height
    )

    t = _phase_time("python init", t_start)

    # build bvh and reorder triangles and materials based on the tree structure
    bvh_nodes, triangles, tri_normals, mat_indices = build_bvh(triangles, tri_normals, mat_indices)
    assert len(bvh_nodes) > 0

    t = _phase_time("bvh build", t)

    # allocate memory for framebuffer and statistics
    fb = cuda.device_array((height, width, 3), dtype=np.uint8)
    out_stats = cuda.device_array((height, width, 2), dtype=np.int32)
    assert out_stats.shape == (height, width, 2)

    t = _phase_time("init cuda alloc", t)

    manager = KernelManager(render_kernel)

    # precompile with use_bvh set to false
    use_bvh = False
    manager.precompile_run(locals())

    threads = (BLOCK_THREADS, BLOCK_THREADS)
    grid = (math.ceil(width / threads[0]), math.ceil(height / threads[1]))
    t = _phase_time("jit compile run", t)

    if RENDER_NON_BVH_STATS:
        # render without data structure (brute force)
        use_bvh = False
        t_brute = manager.run(grid, threads, locals())
        t = _phase_time("render (no ds)", t_brute)

        # fetch and calculate brute force statistics
        stats_brute = out_stats.copy_to_host()
        avg_tris_brute = np.mean(stats_brute[:, :, 0])
        print(f"\t- avg triangle tests per ray (no ds): {avg_tris_brute:.0f}")

        # clear the stats array on the gpu for the next run
        out_stats = cuda.device_array((height, width, 2), dtype=np.int32)

    # render with bvh:
    use_bvh = True
    t_bvh = manager.run(grid, threads, locals())
    t = _phase_time("render (with ds)", t_bvh)

    # fetch and calculate bvh statistics
    stats_bvh = out_stats.copy_to_host()
    avg_tris_bvh = np.mean(stats_bvh[:, :, 0])
    avg_nodes_bvh = np.mean(stats_bvh[:, :, 1])

    print(
        f"\t- normal BVH with avg O(log(n)) * 2 ~ {(np.ceil(np.log2(len(triangles))) * 2):.0f}"
    )  # sometimes testing both branches
    print(f"\t- avg bvh node tests per ray (with ds): {avg_nodes_bvh:.0f}")
    print(f"\t- avg triangle tests per ray (with ds): {avg_tris_bvh:.0f}")

    # retrieve framebuffer from gpu
    host_fb = fb.copy_to_host()

    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "output", "output"
    )
    save_ppm(output_path + ".ppm", host_fb)
    img = Image.fromarray(host_fb)
    img.save(output_path + ".png")
    t = _phase_time("save imgs", t)

    # final total time (+- ending timing output)
    print(f"\n[timing] {'total':<20}: {time.perf_counter() - t_start:7.2f} s")


if __name__ == "__main__":
    main()
