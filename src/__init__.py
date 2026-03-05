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
    USE_BVH_CACHE,
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
    print(
        f"[timing] {label:<20}: {t1 - t0:7.2f} s"
        + (f" ({1.0/(t1 - t0):.1f} FPS)" if fps else "")
    )
    return t1


def count_f64_in_ptx():
    """for float 64 hunting: (and line debug info in the kernel enabled)"""
    ptx_code = render_kernel.inspect_asm()
    for _signature, ptx in ptx_code.items():
        # count double precision instructions
        f64_count = ptx.count(".f64")
        if f64_count > 0:
            print(f"FP64 instructions: {f64_count}")

        # save to file to search for "cvt.f64.f32" (conversion)
        with open("kernel.ptx", "w") as f:
            f.write(ptx)


def load_or_build_scene(json_file, cache_file, t):
    """load scene from cache or build bvh from scratch."""
    if os.path.exists(cache_file) and USE_BVH_CACHE:
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
    return (
        triangles,
        tri_normals,
        mat_indices,
        materials,
        light_data,
        cam_data,
        bvh_nodes,
        t,
    )


def allocate_buffers(width, height):
    """allocate framebuffer and statistics arrays."""
    assert width > 0 and height > 0
    if DEVICE == "gpu":
        fb = cuda.device_array((height, width, 3), dtype=np.uint8)
        out_stats = cuda.device_array((height, width, 5), dtype=np.int32)
    else:
        fb = np.zeros((height, width, 3), dtype=np.uint8)
        out_stats = np.zeros((height, width, 5), dtype=np.int32)

    return fb, out_stats


def print_statistics(stats, render_time, width, height, total_triangles):
    """calculate and print rendering statistics."""
    assert render_time > 0.0

    total_pixels = width * height
    mrays_per_sec = (total_pixels / render_time) / 1_000_000

    avg_primary_tri = np.mean(stats[:, :, 0])
    avg_primary_node = np.mean(stats[:, :, 1])
    optimal_log_nodes = math.log2(total_triangles)
    max_bounces_hit = np.max(stats[:, :, 4])

    print(f"[perf] performance: {mrays_per_sec:.2f} MRays/s")
    print(
        f"[perf] primary node tests: {avg_primary_node:.1f} (O(logN) is ~{optimal_log_nodes:.1f})"
    )
    print(f"[perf] primary triangles tests: {avg_primary_tri:.1f}")
    print(f"[perf] max refraction/reflection depth reached: {max_bounces_hit}")


def save_image(fb, output_path):
    """copy framebuffer to host and save to disk."""
    assert fb is not None
    host_fb = fb.copy_to_host() if DEVICE == "gpu" else fb

    save_ppm(output_path + ".ppm", host_fb)
    img = Image.fromarray(host_fb)
    img.save(output_path + ".jpg")
    print(f"Click to see the result onto: {output_path}.jpg")


def main():
    """run the render pipeline."""
    print(f"Runs on device: {DEVICE.upper()}")
    t = _phase_time("init python", t_start)

    width = int(CPU_DIMENSION) if DEVICE == "cpu" else int(GPU_DIMENSION)
    height = width
    assert width > 0

    json_file = os.path.join(project_root, "box-advanced", "setup.json")
    cache_file_name = json_file.split("/")[-2] + ".bvh.npz"
    cache_file = os.path.join(project_root, "utils", "__pycache__", cache_file_name)

    (
        triangles,
        tri_normals,
        mat_indices,
        materials,
        light_data,
        cam_data,
        bvh_nodes,
        t,
    ) = load_or_build_scene(json_file, cache_file, t)

    origin, p00, qw, qh, light_pos, light_color = build_setup_vectors(
        light_data, cam_data, width, height
    )
    fb, out_stats = allocate_buffers(width, height)
    if DEVICE == "gpu":
        t = _phase_time("init cuda + alloc", t)

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

        stats_brute = out_stats.copy_to_host() if DEVICE == "gpu" else out_stats
        avg_tris_brute = np.mean(stats_brute[:, :, 0])
        print_statistics(stats_brute, t, width, height, len(triangles))
        print()

        # reallocate buffers for the actual run
        fb, out_stats = allocate_buffers(width, height)

    use_bvh = True
    if DEVICE == "gpu":
        cuda.profile_start()
    t_bvh_start = manager.run(grid, threads, locals())
    if DEVICE == "gpu":
        cuda.profile_stop()

    t_bvh_end = _phase_time("render (with ds)", t_bvh_start, fps=True)
    render_time = t_bvh_end - t_bvh_start

    stats = out_stats.copy_to_host() if DEVICE == "gpu" else out_stats
    print_statistics(stats, render_time, width, height, len(triangles))

    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "output", "output"
    )
    save_image(fb, output_path)

    print(f"\n[timing] {'total (+-)':<20}: {time.perf_counter() - t_start:7.2f} s")


# expose 'main' as the only public symbol of the package
__all__ = ["main"]
