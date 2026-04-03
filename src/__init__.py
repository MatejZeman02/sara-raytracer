import sys

print("Importing __init__.py")
sys.stdout.flush()
import sys

print("Importing __init__.py")
sys.stdout.flush()
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
from .constants import (
    BLOCK_THREADS,
    SEED,
    MAT_EMISSIVE_R,
    MAT_EMISSIVE_G,
    MAT_EMISSIVE_B,
)
from .settings import (
    CPU_DIMENSION,
    GPU_DIMENSION,
    DEVICE,
    RENDER_NON_BVH_STATS,
    USE_BVH_CACHE,
    DENOISE,
    PRINT_STATS,
)
from .setup_vectors import build_setup_vectors
from .rng import create_rng_states
from .framebuffer import postprocess_hdr
from .denoiser import denoise

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
    required_cache_keys = (
        "bvh_nodes",
        "triangles",
        "tri_normals",
        "tri_uvs",
        "mat_indices",
        "materials",
        "mat_diffuse_tex_ids",
        "diffuse_textures",
        "tex_widths",
        "tex_heights",
    )

    can_use_cache = False
    if USE_BVH_CACHE and os.path.exists(cache_file):
        cache = np.load(cache_file)
        can_use_cache = all(k in cache.files for k in required_cache_keys)
    if can_use_cache:
        bvh_nodes = cache["bvh_nodes"]
        triangles = cache["triangles"]
        tri_normals = cache["tri_normals"]
        tri_uvs = cache["tri_uvs"]
        mat_indices = cache["mat_indices"]
        materials = cache["materials"]
        mat_diffuse_tex_ids = cache["mat_diffuse_tex_ids"]
        diffuse_textures = cache["diffuse_textures"]
        tex_widths = cache["tex_widths"]
        tex_heights = cache["tex_heights"]
        light_data, cam_data, _ = load_light_cam_data(json_file)
    else:
        (
            triangles,
            tri_normals,
            tri_uvs,
            mat_indices,
            materials,
            mat_diffuse_tex_ids,
            diffuse_textures,
            tex_widths,
            tex_heights,
            light_data,
            cam_data,
        ) = load_scene(json_file)
        assert len(triangles) > 0

        bvh_nodes, triangles, tri_normals, tri_uvs, mat_indices = build_bvh(
            triangles, tri_normals, tri_uvs, mat_indices
        )

        np.savez(
            cache_file,
            bvh_nodes=bvh_nodes,
            triangles=triangles,
            tri_normals=tri_normals,
            tri_uvs=tri_uvs,
            mat_indices=mat_indices,
            materials=materials,
            mat_diffuse_tex_ids=mat_diffuse_tex_ids,
            diffuse_textures=diffuse_textures,
            tex_widths=tex_widths,
            tex_heights=tex_heights,
        )
        t = _phase_time("bvh build", t)
    return (
        triangles,
        tri_normals,
        tri_uvs,
        mat_indices,
        materials,
        mat_diffuse_tex_ids,
        diffuse_textures,
        tex_widths,
        tex_heights,
        light_data,
        cam_data,
        bvh_nodes,
        t,
    )


def allocate_buffers(width, height):
    """allocate HDR framebuffer and statistics arrays."""
    assert width > 0 and height > 0
    # 0: primary tri tests
    # 1: primary node tests
    # 2: primary rays
    # 3: secondary rays (refractions + bounces)
    # 4: shadow rays
    STATS_TUPLE = (height, width, 5)
    FB_HDR_SHAPE = (height, width, 3)  # float32 HDR framebuffer
    if DEVICE == "gpu":
        fb_hdr = cuda.device_array(FB_HDR_SHAPE, dtype=np.float32)
        out_stats = cuda.device_array(STATS_TUPLE, dtype=np.int32)
    else:
        fb_hdr = np.zeros(FB_HDR_SHAPE, dtype=np.float32)
        out_stats = np.zeros(STATS_TUPLE, dtype=np.int32)

    return fb_hdr, out_stats


def print_statistics(stats, render_time, total_triangles, is_ds=True):
    """calculate and print advanced rendering statistics with a focus on readability and ratios."""
    if not PRINT_STATS:
        return
    assert stats.ndim == 3
    assert stats.shape[2] == 5
    assert render_time >= 0.0
    assert total_triangles > 0

    tri_tests = stats[:, :, 0]
    node_tests = stats[:, :, 1]
    prim_rays = stats[:, :, 2]
    sec_rays = stats[:, :, 3]
    shad_rays = stats[:, :, 4]

    # calculate overall sums
    tot_prim = np.sum(prim_rays)
    tot_sec = np.sum(sec_rays)
    tot_shad = np.sum(shad_rays)
    tot_rays = tot_prim + tot_sec + tot_shad

    tot_tri = np.sum(tri_tests)
    tot_node = np.sum(node_tests)
    tot_inc = tot_tri + tot_node

    # calculate performance
    mrays_sec = (tot_rays / 1e6) / render_time if render_time > 0 else 0

    # calculate ratios and averages
    pct_prim = (tot_prim / tot_rays * 100) if tot_rays else 0
    pct_sec = (tot_sec / tot_rays * 100) if tot_rays else 0
    pct_shad = (tot_shad / tot_rays * 100) if tot_rays else 0

    node_tri_ratio = (tot_node / tot_tri) if tot_tri > 0 else 0
    tests_per_ray = (tot_inc / tot_rays) if tot_rays > 0 else 0
    nodes_per_ray = (tot_node / tot_rays) if tot_rays > 0 else 0

    optimal_log_nodes = math.log2(total_triangles)

    # calculate per pixel arrays
    total_rays_per_px = prim_rays + sec_rays + shad_rays
    total_inc_per_px = tri_tests + node_tests

    # calculate pixel workload distribution
    width = stats.shape[1]
    height = stats.shape[0]
    tot_px = width * height

    # empty pixels that only hit sky
    sky_mask = total_rays_per_px == 1
    sky_pixels = np.sum(sky_mask)

    hit_mask = total_rays_per_px > 1
    hit_incidences = total_inc_per_px[hit_mask]

    if len(hit_incidences) > 0:
        mean_inc = np.mean(hit_incidences)
        # pixels needing more than twice the average incidence tests
        hard_thresh = mean_inc * 2.0
        hard_pixels = np.sum(hit_incidences > hard_thresh)
        # standard geometry hits
        standard_pixels = len(hit_incidences) - hard_pixels
    else:
        hard_pixels = 0
        standard_pixels = 0

    pct_sky = (sky_pixels / tot_px) * 100
    pct_std = (standard_pixels / tot_px) * 100
    pct_hard = (hard_pixels / tot_px) * 100

    SEP_LEN = 65
    SEP_EQUAL = "=" * SEP_LEN
    SEP_DASH = "-" * SEP_LEN
    print(
        f"\n{SEP_EQUAL}\n"
        f"  STATISTICS ({'DS' if is_ds else 'No DS'} on {DEVICE.upper()})\n"
        f"{SEP_EQUAL}\n"
        f"Resolution:             {width} x {height} ({tot_prim:,} pixels)\n"
        f"Render time:            {render_time:.3f} s\n"
        f"Throughput (whole run): {mrays_sec:.2f} MRays/s\n"
        f"{SEP_DASH}\n"
        f"RAY DISTRIBUTION (Total: {tot_rays:,})\n"
        f"  Primary:              {pct_prim:5.1f}%  ({tot_prim:,})\n"
        f"  Secondary:            {pct_sec:5.1f}%  ({tot_sec:,})\n"
        f"  Shadow:               {pct_shad:5.1f}%  ({tot_shad:,})\n"
        f"{SEP_DASH}\n"
        f"WORKLOAD DISTRIBUTION\n"
        f"  Sky (1 ray):          {pct_sky:5.1f}%  ({sky_pixels:,})\n"
        f"  Standard geometry:    {pct_std:5.1f}%  ({standard_pixels:,})\n"
        f"  Hard (>> avg tests):  {pct_hard:5.1f}%  ({hard_pixels:,})\n"
        f"{SEP_DASH}\n"
        f"BVH EFFICIENCY (Total incidence ops: {tot_inc:,})\n"
        f"  Node/Triangle ratio:  {node_tri_ratio:.1f} : 1\n"
        f"  Avg ops per ray:      {tests_per_ray:.1f}\n"
        f"  Avg nodes per ray:    {nodes_per_ray:.1f} (Ideal O(logN) ~ {optimal_log_nodes:.1f})\n"
        f"{SEP_DASH}\n"
    )
    if len(hit_incidences):
        print(
            f"PER_HIT-PIXEL LOAD (min / mean / max)\n"
            f"  Rays calls:        {np.min(total_rays_per_px[hit_mask])} / {np.mean(total_rays_per_px[hit_mask]):.1f} / {np.max(total_rays_per_px[hit_mask])}\n"
            f"  Incidence tests:   {np.min(total_inc_per_px[hit_mask])} / {np.mean(total_inc_per_px[hit_mask]):.1f} / {np.max(total_inc_per_px[hit_mask])}\n"
            f"{SEP_EQUAL}"
        )


def save_image(fb, output_path):
    """save uint8 host framebuffer to disk."""
    assert fb is not None
    # fb is always a host numpy uint8 array (produced by postprocess_hdr)
    host_fb = fb.copy_to_host() if hasattr(fb, "copy_to_host") else fb

    # ppm turned off for now...
    # save_ppm(output_path + ".ppm", host_fb)
    img = Image.fromarray(host_fb)
    EXT = ".jpg"
    img.save(output_path + EXT)
    print(f"Click to see the result onto: {output_path}{EXT}")


def main():
    """run the render pipeline."""
    print(f"Runs on device: {DEVICE.upper()}")
    t = _phase_time("init python", t_start)

    width = int(CPU_DIMENSION) if DEVICE == "cpu" else int(GPU_DIMENSION)
    height = width
    assert width > 0

    json_file = os.path.join(project_root, "scenes", "box-advanced", "setup.json")
    cache_file_name = json_file.split("/")[-2] + ".bvh.npz"
    cache_file = os.path.join(project_root, "utils", "__pycache__", cache_file_name)

    (
        triangles,
        tri_normals,
        tri_uvs,
        mat_indices,
        materials,
        mat_diffuse_tex_ids,
        diffuse_textures,
        tex_widths,
        tex_heights,
        light_data,
        cam_data,
        bvh_nodes,
        t,
    ) = load_or_build_scene(json_file, cache_file, t)

    origin, p00, qw, qh, light_pos, light_color = build_setup_vectors(
        light_data, cam_data, width, height
    )
    fb_hdr, out_stats = allocate_buffers(width, height)
    if DEVICE == "gpu":
        t = _phase_time("init cuda + alloc", t)

    # build index array of emissive triangles (after bvh reordering) for area light sampling
    emissive_mask = (
        (materials[mat_indices, MAT_EMISSIVE_R] > 0)
        | (materials[mat_indices, MAT_EMISSIVE_G] > 0)
        | (materials[mat_indices, MAT_EMISSIVE_B] > 0)
    )
    emissive_tris = np.where(emissive_mask)[0].astype(np.int32)
    num_emissive = np.int32(len(emissive_tris))
    assert (
        num_emissive > 0
    ), "scene has no emissive triangles - area light requires at least one"

    # per-pixel rng states sized for the full image
    rng_states = create_rng_states(width * height, seed=SEED)

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
        print_statistics(stats_brute, t - t_brute, len(triangles), is_ds=use_bvh)
        print()

        # reallocate buffers for the actual run
        fb_hdr, out_stats = allocate_buffers(width, height)

    use_bvh = False
    if DEVICE == "gpu":
        cuda.profile_start()
    t_bvh_start = manager.run(grid, threads, locals())
    if DEVICE == "gpu":
        cuda.profile_stop()

    t_bvh_end = _phase_time("render (with ds)", t_bvh_start, fps=True)
    render_time = t_bvh_end - t_bvh_start

    # copy HDR buffer to host, denoise, then apply ACES+sRGB to produce uint8
    fb_hdr_host = fb_hdr.copy_to_host() if DEVICE == "gpu" else fb_hdr
    t = _phase_time("copy hdr to host", t_bvh_end)
    if DENOISE:
        denoise(fb_hdr_host, width, height)
        t = _phase_time("oidn denoise", t)

    fb = np.zeros((height, width, 3), dtype=np.uint8)
    postprocess_hdr(fb_hdr_host, fb, width, height)
    t = _phase_time("postprocess (srgb/tonemapper on CPU)", t)
    # add postprocess time to render time for more accurate "total time to final image" stat
    render_time += t - t_bvh_end

    stats = out_stats.copy_to_host() if DEVICE == "gpu" else out_stats
    print_statistics(stats, render_time, len(triangles))

    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "output", "output"
    )
    save_image(fb, output_path)

    print(f"\n[timing] {'total (+-)':<20}: {time.perf_counter() - t_start:7.2f} s")


# expose 'main' as the only public symbol of the package
__all__ = ["main"]
