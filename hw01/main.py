"""Homework 01 renderer entrypoint."""

# system lib imports:
import os
import sys
import time
import math

# 3rd party imports:
import numpy as np  # type: ignore
from numba import cuda  # type: ignore
from PIL import Image  # type: ignore

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


def main():
    """
    Main function that orchestrates the ray tracing rendering pipeline.

    This function loads a 3D scene from a JSON configuration file, transfers geometry
    and material data to GPU memory, sets up camera parameters, and executes a CUDA
    kernel to render the scene. The resulting framebuffer is then saved as both PPM
    and PNG image files.

    The function performs the following steps:
    1. Initializes render dimensions based on DIMENSION constant
    2. Loads scene data (triangles, materials, camera configuration) from JSON
    3. Transfers all geometry and material data to GPU device memory (d_* variables)
    4. Builds camera coordinate system vectors from camera data
    5. Configures CUDA thread blocks and grid dimensions for parallel execution
    6. Executes the render_kernel on GPU to perform ray tracing
    7. Copies rendered framebuffer from GPU back to host memory
    8. Saves the output as both PPM and PNG formats
    9. Reports render execution time
    """
    width, height = int(DIMENSION), int(DIMENSION)

    json_file = os.path.join(project_root, "box-advanced", "setup.json")
    triangles, mat_indices, materials, cam_data = load_scene(json_file)

    d_triangles = cuda.to_device(triangles)
    d_mat_indices = cuda.to_device(mat_indices)
    d_materials = cuda.to_device(materials)

    origin, p00, q_w, q_h, light_pos, light_color = build_camera_vectors(
        cam_data, width, height
    )

    d_p00 = cuda.to_device(np.array(p00, dtype=np.float32))
    d_qw = cuda.to_device(np.array(q_w, dtype=np.float32))
    d_qh = cuda.to_device(np.array(q_h, dtype=np.float32))
    d_origin = cuda.to_device(origin)
    d_light_pos = cuda.to_device(light_pos)
    d_light_color = cuda.to_device(light_color)

    d_fb = cuda.device_array((height, width, 3), dtype=np.uint8)

    threads_per_block = (BLOCK_THREADS, BLOCK_THREADS)
    blocks_per_grid_x = math.ceil(width / threads_per_block[0])
    blocks_per_grid_y = math.ceil(height / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    start_time = time.time()
    render_kernel[blocks_per_grid, threads_per_block](
        d_triangles,
        d_mat_indices,
        d_materials,
        d_light_pos,
        d_light_color,
        d_p00,
        d_qw,
        d_qh,
        d_origin,
        d_fb,
        width,
        height,
    )

    host_fb = d_fb.copy_to_host()

    cuda.synchronize()
    end_time = time.time()

    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "output", "output"
    )
    save_ppm(output_path + ".ppm", host_fb)
    img = Image.fromarray(host_fb)
    img.save(output_path + ".png")

    print(f"Render complete in {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
