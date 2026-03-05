"""Homework renderer package entrypoint."""

# pylint: disable=too-many-locals, unused-variable, line-too-long

import time

LOAD_PYTHON_TIME = 1.0
t_start = time.perf_counter() - LOAD_PYTHON_TIME

import math
import os
import sys
import warnings

import numpy as np
from numba.core.errors import NumbaPerformanceWarning
from PIL import Image

from .bvh import build_bvh
from .render_kernel import render_kernel
from .constants import BLOCK_THREADS
from .settings import (
    CPU_DIMENSION,
    GPU_DIMENSION,
    DEVICE,
    RENDER_NON_BVH_STATS,
    USE_CACHE,
)
from .setup_vectors import build_setup_vectors

if DEVICE == "gpu":
    from numba import cuda

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.kernel_manager import KernelManager
from utils.obj_loader import load_light_cam_data, load_scene
from utils.ppm import save_ppm


def _phase_time(label: str, t0: float, fps: bool = False) -> float:
    """Print elapsed time from t0 and return new timestamp."""
    t1 = time.perf_counter()
    if fps:
        print(f"[timing] {label:<20}: {t1 - t0:7.2f} s ({1.0/(t1 - t0):.1f} FPS)")
    else:
        print(f"[timing] {label:<20}: {t1 - t0:7.2f} s")
    return t1


def main():
    """Run the render pipeline."""
    print(f"Runs on device: {DEVICE.upper()}")
    t = _phase_time("init python", t_start)
    width, height = int(GPU_DIMENSION), int(GPU_DIMENSION)
    if DEVICE == "cpu":
        width, height = int(CPU_DIMENSION), int(CPU_DIMENSION)

    # json_file = os.path.join(project_root, "box-sphere-original", "setup.json")
    json_file = os.path.join(project_root, "box-advanced", "setup.json")
    cache_file_name = json_file.split("/")[-2] + ".bvh.npz"
    cache_file = os.path.join(project_root, "utils", "__pycache__", cache_file_name)

    if os.path.exists(cache_file) and USE_CACHE:
        cache = np.load(cache_file)
        bvh_nodes = cache["bvh_nodes"]
        triangles = cache["triangles"]
        tri_normals = cache["tri_normals"]
        mat_indices = cache["mat_indices"]
        materials = cache["materials"]
        light_data, cam_data, _ = load_light_cam_data(json_file)
    else:
        triangles, tri_normals, mat_indices, materials, light_data, cam_data = (
            load_scene(json_file)
        )
        assert len(triangles) > 0

        bvh_nodes, triangles, tri_normals, mat_indices = build_bvh(
            triangles, tri_normals, mat_indices
        )

        np.savez(
            cache_file,
            bvh_nodes=bvh_nodes,
            triangles=triangles,
            tri_normals=tri_normals,
            mat_indices=mat_indices,
            materials=materials,
        )
        t = _phase_time("bvh build", t)

    origin, p00, qw, qh, light_pos, light_color = build_setup_vectors(
        light_data, cam_data, width, height
    )

    # allocate framebuffer and statistics arrays
    if DEVICE == "gpu":
        fb = cuda.device_array((height, width, 3), dtype=np.uint8)
        out_stats = cuda.device_array((height, width, 2), dtype=np.int32)
    else:
        fb = np.zeros((height, width, 3), dtype=np.uint8)
        out_stats = np.zeros((height, width, 2), dtype=np.int32)
    assert out_stats.shape == (height, width, 2)

    t = _phase_time("init alloc", t)

    manager = KernelManager(render_kernel)
    use_bvh = False
    manager.precompile_run(locals())

    threads = (BLOCK_THREADS, BLOCK_THREADS)
    grid = (math.ceil(width / threads[0]), math.ceil(height / threads[1]))
    t = _phase_time("jit compile run", t)

    if RENDER_NON_BVH_STATS:
        use_bvh = False
        t_brute = manager.run(grid, threads, locals())
        t = _phase_time("render (no ds)", t_brute)

        if DEVICE == "gpu":
            stats_brute = out_stats.copy_to_host()
        else:
            stats_brute = out_stats
        avg_tris_brute = np.mean(stats_brute[:, :, 0])
        print(f"\t- avg triangle tests per ray (no ds): {avg_tris_brute:.0f}")
        if DEVICE == "gpu":
            out_stats = cuda.device_array((height, width, 2), dtype=np.int32)
        else:
            out_stats = np.zeros((height, width, 2), dtype=np.int32)

    use_bvh = True
    if DEVICE == "gpu":
        cuda.profile_start()
    t_bvh = manager.run(grid, threads, locals())
    if DEVICE == "gpu":
        cuda.profile_stop()
    _phase_time("render (with ds)", t_bvh, fps=True)

    if DEVICE == "gpu":
        stats_bvh = out_stats.copy_to_host()
    else:
        stats_bvh = out_stats
    avg_tris_bvh = np.mean(stats_bvh[:, :, 0])
    avg_nodes_bvh = np.mean(stats_bvh[:, :, 1])

    # print(
    #     f"\t- normal BVH with avg O(log(n)) * 2 ~ {(np.ceil(np.log2(len(triangles))) * 2):.0f}"
    # )
    print(f"\t- avg bvh node tests per ray (with ds): {avg_nodes_bvh:.0f}")
    print(f"\t- avg triangle tests per ray (with ds): {avg_tris_bvh:.0f}")

    if DEVICE == "gpu":
        host_fb = fb.copy_to_host()
    else:
        host_fb = fb

    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "output", "output"
    )
    save_ppm(output_path + ".ppm", host_fb)
    img = Image.fromarray(host_fb)
    img.save(output_path + ".png")
    _phase_time("save imgs", t)

    print(f"\n[timing] {'total':<20}: {time.perf_counter() - t_start:7.2f} s")
    print("Click to see the result onto: src/output/output.png")

    # for float 64 hunting: (and line debug info in the kernel enabled)
    # # print the compiled assembly
    # ptx_code = render_kernel.inspect_asm()
    # for signature, ptx in ptx_code.items():
    #     # count double precision instructions
    #     f64_count = ptx.count(".f64")
    #     if f64_count > 0:
    #         print(f"FP64 instructions: {f64_count}")

    #     # save to file to search for "cvt.f64.f32" (conversion)
    #     with open("kernel.ptx", "w") as f:
    #         f.write(ptx)


# expose 'main' as the only public symbol of the package
__all__ = ["main"]
