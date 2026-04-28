"""Homework renderer package entrypoint."""

# pylint: disable=too-many-locals, unused-variable, line-too-long

import time

LOAD_PYTHON_TIME = 1.3  # meassured using 'time' command on empty main().
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
from .wavefront_kernel import render_kernel_pass1, render_kernel_pass2
from .constants import (
    BLOCK_THREADS,
    SEED,
    MAT_EMISSIVE_R,
    MAT_EMISSIVE_G,
    MAT_EMISSIVE_B,
    PRIMARY_RAY,
    SECONDARY_RAY,
    SHADOW_RAY,
    WF_STATUS_ACTIVE,
)
from .settings import (
    CPU_DIMENSION,
    GPU_DIMENSION,
    DEVICE,
    EXECUTION_MODE,
    CPU_PARALLEL,
    GPU_BLOCK_X,
    GPU_BLOCK_Y,
    RENDER_NON_BVH_STATS,
    USE_BVH_CACHE,
    DENOISE,
    PRINT_STATS,
    IMG_FORMAT,
    SCENE_NAME,
    WAVEFRONT_ENABLED,
    BVH_OPS_BUDGET,
)
from .setup_vectors import build_setup_vectors
from .rng import create_rng_states
from .framebuffer import postprocess_hdr
from .denoiser import HAS_OIDN, denoise

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


def run_render_timed(manager, grid, threads, data_context):
    """Run the render kernel and return strict elapsed wall-clock time."""
    t0 = time.perf_counter()
    manager.run(grid, threads, data_context, measure_time=False)
    t1 = time.perf_counter()
    return t1 - t0


def _to_device_array(value):
    """Move numpy arrays to the active CUDA device; keep scalars as-is."""
    if hasattr(value, "copy_to_host"):
        return value
    if isinstance(value, np.ndarray):
        return cuda.to_device(value)
    return value


def allocate_wavefront_state(num_pixels):
    """Allocate global ray-state buffers used for stream compaction passes."""
    assert DEVICE == "gpu"
    assert num_pixels > 0

    zeros_vec = np.zeros((num_pixels, 3), dtype=np.float32)
    zeros_i32 = np.zeros(num_pixels, dtype=np.int32)
    ones_i32 = np.ones(num_pixels, dtype=np.int32)

    return {
        "ray_o": cuda.to_device(zeros_vec.copy()),
        "ray_d": cuda.to_device(zeros_vec.copy()),
        "accum_color": cuda.to_device(zeros_vec.copy()),
        "sample_color": cuda.to_device(zeros_vec.copy()),
        "throughput": cuda.to_device(zeros_vec.copy()),
        "current_sample": cuda.to_device(zeros_i32.copy()),
        "current_bounce": cuda.to_device(zeros_i32.copy()),
        "in_path": cuda.to_device(zeros_i32.copy()),
        "status": cuda.to_device(ones_i32),
    }


def run_wavefront_render_timed(
    scene,
    fb_hdr,
    out_stats,
    width,
    height,
    rng_states,
    emissive_tris,
    num_emissive,
):
    """Execute wavefront rendering in two passes with host-side compaction telemetry."""
    assert DEVICE == "gpu"
    assert render_kernel_pass1 is not None and render_kernel_pass2 is not None

    width_i32 = np.int32(width)
    height_i32 = np.int32(height)
    num_pixels = int(width_i32) * int(height_i32)

    state = allocate_wavefront_state(num_pixels)

    threads_2d = (GPU_BLOCK_X, GPU_BLOCK_Y)
    grid_2d = (
        math.ceil(int(width_i32) / threads_2d[0]),
        math.ceil(int(height_i32) / threads_2d[1]),
    )

    pass1_t0 = time.perf_counter()
    render_kernel_pass1[grid_2d, threads_2d](
        scene["triangles"],
        scene["tri_normals"],
        scene["tri_uvs"],
        scene["mat_indices"],
        scene["materials"],
        scene["mat_diffuse_tex_ids"],
        scene["diffuse_textures"],
        scene["tex_widths"],
        scene["tex_heights"],
        scene["bvh_nodes"],
        scene["use_bvh"],
        scene["p00"],
        scene["qw"],
        scene["qh"],
        scene["origin"],
        fb_hdr,
        out_stats,
        width_i32,
        height_i32,
        rng_states,
        emissive_tris,
        num_emissive,
        np.int32(BVH_OPS_BUDGET),
        state["ray_o"],
        state["ray_d"],
        state["accum_color"],
        state["sample_color"],
        state["throughput"],
        state["current_sample"],
        state["current_bounce"],
        state["in_path"],
        state["status"],
    )
    cuda.synchronize()
    pass1_time = time.perf_counter() - pass1_t0

    compaction_t0 = time.perf_counter()
    status_host = state["status"].copy_to_host()
    active_idx_host = np.where(status_host == WF_STATUS_ACTIVE)[0].astype(np.int32)
    compaction_time = time.perf_counter() - compaction_t0

    pass2_time = 0.0
    active_count = int(active_idx_host.size)
    if active_count > 0:
        active_idx_dev = cuda.to_device(active_idx_host)
        threads_1d = 256
        grid_1d = math.ceil(active_count / threads_1d)

        pass2_t0 = time.perf_counter()
        render_kernel_pass2[grid_1d, threads_1d](
            active_idx_dev,
            np.int32(active_count),
            scene["triangles"],
            scene["tri_normals"],
            scene["tri_uvs"],
            scene["mat_indices"],
            scene["materials"],
            scene["mat_diffuse_tex_ids"],
            scene["diffuse_textures"],
            scene["tex_widths"],
            scene["tex_heights"],
            scene["bvh_nodes"],
            scene["use_bvh"],
            scene["p00"],
            scene["qw"],
            scene["qh"],
            scene["origin"],
            fb_hdr,
            out_stats,
            width_i32,
            height_i32,
            rng_states,
            emissive_tris,
            num_emissive,
            state["ray_o"],
            state["ray_d"],
            state["accum_color"],
            state["sample_color"],
            state["throughput"],
            state["current_sample"],
            state["current_bounce"],
            state["in_path"],
            state["status"],
        )
        cuda.synchronize()
        pass2_time = time.perf_counter() - pass2_t0

    total_time = pass1_time + compaction_time + pass2_time
    return {
        "pass1_render_s": pass1_time,
        "cpu_compaction_s": compaction_time,
        "pass2_render_s": pass2_time,
        "total_render_s": total_time,
        "active_rays_after_pass1": active_count,
    }


def calculate_render_metrics(stats, render_time):
    """Calculate baseline metrics: total rays and throughput in MRays/s."""
    assert stats.ndim == 3
    assert stats.shape[2] == 5
    total_primary = int(np.sum(stats[:, :, PRIMARY_RAY], dtype=np.int64))
    total_secondary = int(np.sum(stats[:, :, SECONDARY_RAY], dtype=np.int64))
    total_shadow = int(np.sum(stats[:, :, SHADOW_RAY], dtype=np.int64))
    total_rays = total_primary + total_secondary + total_shadow
    mrays_per_sec = (total_rays / 1e6) / render_time if render_time > 0.0 else 0.0
    return {
        "render_time_s": render_time,
        "primary_rays": total_primary,
        "secondary_rays": total_secondary,
        "shadow_rays": total_shadow,
        "total_rays": total_rays,
        "mrays_per_sec": mrays_per_sec,
    }


def print_render_metrics(metrics, label="with ds"):
    """Print strict baseline throughput metrics."""
    print(
        f"[metrics] {label:<20}: {metrics['render_time_s']:7.4f} s\n"
        f"[metrics] total rays cast      : {metrics['total_rays']:,}"
        f" (primary={metrics['primary_rays']:,}, secondary={metrics['secondary_rays']:,}, shadow={metrics['shadow_rays']:,})\n"
        f"[metrics] throughput           : {metrics['mrays_per_sec']:.3f} MRays/s"
    )


def count_f64_in_ptx():
    """for float 64 hunting (and line debug info in the kernel enabled)"""
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
    img.save(f"{output_path}.{IMG_FORMAT}")
    print(f"Click to see the result onto: {output_path}.{IMG_FORMAT}")


def main():
    """run the render pipeline."""
    if DEVICE == "cpu":
        cpu_mode = "parallel" if CPU_PARALLEL else "sequential"
        print(f"Runs on device: CPU ({cpu_mode})")
    else:
        print(f"Runs on device: GPU ({GPU_BLOCK_X}x{GPU_BLOCK_Y} threads/block)")
    print(f"[config] execution mode      : {EXECUTION_MODE}")
    print(f"[config] wavefront enabled   : {WAVEFRONT_ENABLED}")
    if WAVEFRONT_ENABLED:
        print(f"[config] bvh ops budget      : {BVH_OPS_BUDGET}")
    t = _phase_time("init python", t_start)

    width_host = int(CPU_DIMENSION) if DEVICE == "cpu" else int(GPU_DIMENSION)
    height_host = width_host
    assert width_host > 0

    json_file = os.path.join(project_root, "scenes", SCENE_NAME, "setup.json")
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
        light_data, cam_data, width_host, height_host
    )
    fb_hdr, out_stats = allocate_buffers(width_host, height_host)
    if DEVICE == "gpu":
        t = _phase_time("init cuda + alloc", t)

    # pass explicit int32 dimensions to kernels to avoid mixed scalar typing.
    width = np.int32(width_host)
    height = np.int32(height_host)

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
    rng_count = np.int32(width) * np.int32(height)
    rng_states = create_rng_states(int(rng_count), seed=int(np.uint64(SEED)))

    use_wavefront = DEVICE == "gpu" and WAVEFRONT_ENABLED
    use_bvh = True
    wavefront_timings = None

    if use_wavefront:
        if render_kernel_pass1 is None or render_kernel_pass2 is None:
            raise RuntimeError("Wavefront kernels are unavailable in current mode")

        scene_wavefront = {
            "triangles": _to_device_array(triangles),
            "tri_normals": _to_device_array(tri_normals),
            "tri_uvs": _to_device_array(tri_uvs),
            "mat_indices": _to_device_array(mat_indices),
            "materials": _to_device_array(materials),
            "mat_diffuse_tex_ids": _to_device_array(mat_diffuse_tex_ids),
            "diffuse_textures": _to_device_array(diffuse_textures),
            "tex_widths": _to_device_array(tex_widths),
            "tex_heights": _to_device_array(tex_heights),
            "bvh_nodes": _to_device_array(bvh_nodes),
            "use_bvh": use_bvh,
            "p00": _to_device_array(p00),
            "qw": _to_device_array(qw),
            "qh": _to_device_array(qh),
            "origin": _to_device_array(origin),
        }
        emissive_tris_dev = _to_device_array(emissive_tris)

        # Warmup launch to compile both passes and keep render telemetry clean.
        warmup_fb = cuda.device_array((1, 1, 3), dtype=np.float32)
        warmup_stats = cuda.device_array((1, 1, 5), dtype=np.int32)
        warmup_rng = create_rng_states(1, seed=int(np.uint64(SEED)))
        warmup_state = allocate_wavefront_state(1)

        render_kernel_pass1[(1, 1), (1, 1)](
            scene_wavefront["triangles"],
            scene_wavefront["tri_normals"],
            scene_wavefront["tri_uvs"],
            scene_wavefront["mat_indices"],
            scene_wavefront["materials"],
            scene_wavefront["mat_diffuse_tex_ids"],
            scene_wavefront["diffuse_textures"],
            scene_wavefront["tex_widths"],
            scene_wavefront["tex_heights"],
            scene_wavefront["bvh_nodes"],
            True,
            scene_wavefront["p00"],
            scene_wavefront["qw"],
            scene_wavefront["qh"],
            scene_wavefront["origin"],
            warmup_fb,
            warmup_stats,
            np.int32(1),
            np.int32(1),
            warmup_rng,
            emissive_tris_dev,
            num_emissive,
            np.int32(1),
            warmup_state["ray_o"],
            warmup_state["ray_d"],
            warmup_state["accum_color"],
            warmup_state["sample_color"],
            warmup_state["throughput"],
            warmup_state["current_sample"],
            warmup_state["current_bounce"],
            warmup_state["in_path"],
            warmup_state["status"],
        )

        warmup_active = cuda.to_device(np.zeros(1, dtype=np.int32))
        render_kernel_pass2[1, 1](
            warmup_active,
            np.int32(0),
            scene_wavefront["triangles"],
            scene_wavefront["tri_normals"],
            scene_wavefront["tri_uvs"],
            scene_wavefront["mat_indices"],
            scene_wavefront["materials"],
            scene_wavefront["mat_diffuse_tex_ids"],
            scene_wavefront["diffuse_textures"],
            scene_wavefront["tex_widths"],
            scene_wavefront["tex_heights"],
            scene_wavefront["bvh_nodes"],
            True,
            scene_wavefront["p00"],
            scene_wavefront["qw"],
            scene_wavefront["qh"],
            scene_wavefront["origin"],
            warmup_fb,
            warmup_stats,
            np.int32(1),
            np.int32(1),
            warmup_rng,
            emissive_tris_dev,
            num_emissive,
            warmup_state["ray_o"],
            warmup_state["ray_d"],
            warmup_state["accum_color"],
            warmup_state["sample_color"],
            warmup_state["throughput"],
            warmup_state["current_sample"],
            warmup_state["current_bounce"],
            warmup_state["in_path"],
            warmup_state["status"],
        )
        cuda.synchronize()
    else:
        manager = KernelManager(render_kernel)
        manager.precompile_run(locals())

        if DEVICE == "gpu":
            threads = (GPU_BLOCK_X, GPU_BLOCK_Y)
            grid = (
                math.ceil(int(width) / threads[0]),
                math.ceil(int(height) / threads[1]),
            )
        else:
            # CPU path ignores grid and block in KernelManager.
            threads = (BLOCK_THREADS, BLOCK_THREADS)
            grid = (1, 1)

    t = _phase_time("jit compile run", t)

    if RENDER_NON_BVH_STATS and not use_wavefront:
        use_bvh = False
        brute_render_time = run_render_timed(manager, grid, threads, locals())
        brute_fps = (1.0 / brute_render_time) if brute_render_time > 0.0 else 0.0
        print(
            f"[timing] {'render (no ds)':<20}: {brute_render_time:7.2f} s ({brute_fps:.1f} FPS)"
        )

        stats_brute = out_stats.copy_to_host() if DEVICE == "gpu" else out_stats
        print_render_metrics(
            calculate_render_metrics(stats_brute, brute_render_time),
            label="no ds",
        )
        print_statistics(stats_brute, brute_render_time, len(triangles), is_ds=False)
        print()

        # reallocate buffers for the actual run
        fb_hdr, out_stats = allocate_buffers(width_host, height_host)
    elif RENDER_NON_BVH_STATS and use_wavefront:
        warnings.warn(
            "RENDER_NON_BVH_STATS is ignored in wavefront mode.",
            RuntimeWarning,
        )

    use_bvh = True
    if DEVICE == "gpu":
        cuda.profile_start()

    if use_wavefront:
        scene_wavefront["use_bvh"] = use_bvh
        wavefront_timings = run_wavefront_render_timed(
            scene_wavefront,
            fb_hdr,
            out_stats,
            width,
            height,
            rng_states,
            emissive_tris_dev,
            num_emissive,
        )
        kernel_render_time = wavefront_timings["total_render_s"]
    else:
        kernel_render_time = run_render_timed(manager, grid, threads, locals())

    if DEVICE == "gpu":
        cuda.profile_stop()

    if wavefront_timings is not None:
        print(
            f"[timing] {'pass1 render':<20}: {wavefront_timings['pass1_render_s']:7.4f} s"
        )
        print(
            f"[timing] {'cpu compaction':<20}: {wavefront_timings['cpu_compaction_s']:7.4f} s"
        )
        print(
            f"[timing] {'pass2 render':<20}: {wavefront_timings['pass2_render_s']:7.4f} s"
        )
        print(
            f"[timing] {'active rays after p1':<20}: {wavefront_timings['active_rays_after_pass1']}"
        )

    render_fps = (1.0 / kernel_render_time) if kernel_render_time > 0.0 else 0.0
    print(
        f"[timing] {'render (with ds)':<20}: {kernel_render_time:7.2f} s ({render_fps:.1f} FPS)"
    )
    t_bvh_end = time.perf_counter()
    render_time = kernel_render_time

    # copy HDR buffer to host, denoise, then apply ACES+sRGB to produce uint8
    fb_hdr_host = fb_hdr.copy_to_host() if DEVICE == "gpu" else fb_hdr
    t = _phase_time("copy hdr to host", t_bvh_end)
    if DENOISE and HAS_OIDN:
        denoise(fb_hdr_host, width_host, height_host)
        t = _phase_time("oidn denoise", t)
    elif DENOISE:
        warnings.warn(
            "DENOISE is enabled but OIDN is not installed; skipping denoising.",
            RuntimeWarning,
        )

    fb = np.zeros((height_host, width_host, 3), dtype=np.uint8)
    postprocess_hdr(fb_hdr_host, fb, width_host, height_host)
    t = _phase_time("postprocess (srgb/tonemapper on CPU)", t)
    # add postprocess time to render time for more accurate "total time to final image" stat
    render_time += t - t_bvh_end

    stats = out_stats.copy_to_host() if DEVICE == "gpu" else out_stats
    print_render_metrics(calculate_render_metrics(stats, kernel_render_time))
    print_statistics(stats, render_time, len(triangles))

    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "output", "output"
    )
    save_image(fb, output_path)

    print(f"\n[timing] {'total (+-)':<20}: {time.perf_counter() - t_start:7.2f} s")


# expose 'main' as the only public symbol of the package
__all__ = ["main"]
