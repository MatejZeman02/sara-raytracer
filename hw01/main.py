"""Homework 01 renderer entrypoint."""

# pylint: disable=too-many-locals, unused-variable, line-too-long

# system lib imports:
import time

# global start time for total execution timing
t_start = time.perf_counter()

import os
import sys
import warnings
import math

# 3rd party imports:
import numpy as np  # type: ignore
from PIL import Image  # type: ignore
from numba import cuda  # type: ignore
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
from settings import BLOCK_THREADS, DIMENSION
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
    json_file = os.path.join(project_root, "box-advanced", "setup.json")
    triangles, mat_indices, materials, cam_data = load_scene(json_file)

    origin, p00, qw, qh, light_pos, light_color = build_camera_vectors(
        cam_data, width, height
    )

    t = _phase_time("init", t_start)
    fb = cuda.device_array((height, width, 3), dtype=np.uint8)
    t = _phase_time("init cuda alloc fb", t)

    manager = KernelManager(render_kernel)
    manager.precompile_run(locals())  # compile/warmup
    cuda.synchronize()

    threads = (BLOCK_THREADS, BLOCK_THREADS)
    grid = (math.ceil(width / threads[0]), math.ceil(height / threads[1]))

    t = _phase_time("jit compile run", t)
    # run kernel using current all local variables
    # the manager sees 'triangles' and maps it to 'd_triangles'
    manager.run(grid, threads, locals())
    cuda.synchronize()
    t = _phase_time("kernel run", t)

    # retrieve framebuffer from gpu
    host_fb = fb.copy_to_host()

    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "output", "output"
    )
    save_ppm(output_path + ".ppm", host_fb)
    img = Image.fromarray(host_fb)
    img.save(output_path + ".png")
    t = _phase_time("save", t)

    print(f"\n[timing] {'total':<20}: {time.perf_counter() - t_start:7.2f} s")


if __name__ == "__main__":
    main()
