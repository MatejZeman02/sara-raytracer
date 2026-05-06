"""Homework renderer package entrypoint."""

import time
import math
import os
import sys
import warnings

LOAD_PYTHON_TIME = 1.3
t_start = time.perf_counter() - LOAD_PYTHON_TIME

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
from numba.core.errors import NumbaPerformanceWarning
from PIL import Image

# Core rendering
from .render_kernel import render_kernel, collect_bvh_stats
from .constants import (
    BLOCK_THREADS,
    SEED,
    MAT_EMISSIVE_R,
    MAT_EMISSIVE_G,
    MAT_EMISSIVE_B,
)
from .settings import settings
from .setup_vectors import build_setup_vectors
from .rng import create_rng_states
from .framebuffer import (
    postprocess_sdr_to_u8,
    postprocess_full_gpu_kernel,
    tonemap_hdr_to_sdr,
    tonemap_kernel,
    create_gamma_lut,
)

# Refactored modules
from ._scene import load_or_build_scene
from ._buffers import allocate_buffers
from ._stats import print_statistics
from ._io import save_image
from ._denoise import denoise_gpu_hdr, denoise_cpu_ldr, get_denoise_path
from utils.smart_denoiser import HAS_NATIVE_CUDA_OIDN, HAS_PIP_OIDN
from utils.kernel_manager import KernelManager
from utils.obj_loader import load_light_cam_data

if settings.DEVICE == "gpu":
    from numba import cuda

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

project_root = _project_root


def _phase_time(label: str, t0: float, fps: bool = False) -> float:
    """Print elapsed time from t0 and return new timestamp."""
    t1 = time.perf_counter()
    print(
        f"[timing] {label:<20}: {t1 - t0:7.2f} s"
        + (f" ({1.0/(t1 - t0):.1f} FPS)" if fps else "")
    )
    return t1


def main():
    """Run the render pipeline."""
    # Handle --collect-bvh-stats flag
    collect_stats = settings.COLLECT_BVH_STATS
    for arg in sys.argv[1:]:
        if arg == "--collect-bvh-stats":
            collect_stats = True

    if settings.DEVICE == "gpu":
        print(f"Runs on device: {str(cuda.get_current_device().name)[2:-1]}", end="")
    else:
        print(f"Runs on device: {settings.DEVICE.upper()}")
    print(
        f", scene: {settings.SCENE_NAME}, samples: {settings.SAMPLES}, max bounces: {settings.MAX_BOUNCES}, tonemapper: {settings.TONEMAPPER}"
    )
    t = _phase_time("init python", t_start)

    width_host = (
        int(settings.CPU_DIMENSION)
        if settings.DEVICE == "cpu"
        else int(settings.GPU_DIMENSION)
    )
    height_host = width_host
    assert width_host > 0

    json_file = os.path.join(project_root, "scenes", settings.SCENE_NAME, "setup.json")
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
        _,
        base_exposure,
        bvh_build_time,
    ) = load_or_build_scene(json_file, cache_file, t)
    if bvh_build_time > 0:
        t = _phase_time("bvh build", t - bvh_build_time)
    else:
        t = time.perf_counter()

    origin, p00, qw, qh, light_pos, light_color = build_setup_vectors(
        light_data, cam_data, width_host, height_host
    )
    fb_hdr, out_stats = allocate_buffers(width_host, height_host)
    if settings.DEVICE == "gpu":
        t = _phase_time("init cuda + alloc", t)

    width = np.int32(width_host)
    height = np.int32(height_host)

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

    rng_count = np.int32(width) * np.int32(height)
    rng_states = create_rng_states(int(rng_count), seed=int(np.uint64(SEED)))

    metrics_out = (
        cuda.device_array((int(width_host) * int(height_host), 4), dtype=np.float32)
        if settings.DEVICE == "gpu"
        else np.zeros((int(width_host) * int(height_host), 4), dtype=np.float32)
    )

    manager = KernelManager(render_kernel)
    use_bvh = True
    manager.precompile_run(locals())

    threads = (BLOCK_THREADS, BLOCK_THREADS)
    grid = (
        math.ceil(int(width) / threads[0]),
        math.ceil(int(height) / threads[1]),
    )
    t = _phase_time("jit compile run", t)

    # If collecting BVH stats, run metrics collection instead of full render
    if collect_stats:
        output_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "output", "bvh_stats.txt"
        )
        collect_bvh_stats(
            triangles,
            tri_normals,
            tri_uvs,
            mat_indices,
            materials,
            mat_diffuse_tex_ids,
            diffuse_textures,
            tex_widths,
            tex_heights,
            bvh_nodes,
            p00,
            qw,
            qh,
            origin,
            width_host,
            height_host,
            rng_states,
            emissive_tris,
            num_emissive,
            output_path,
        )
        print(f"\n[timing] {'total (+-)':<20}: {time.perf_counter() - t_start:7.2f} s")
        return

    # Calculate camera exposure multiplier once on CPU
    # base_exposure from setup.json scales scene light level
    exposure_mul = (
        float(np.float32(2.0**settings.EXPOSURE_COMPENSATION)) * base_exposure
    )

    # Warmup postprocess kernels
    gamma_lut = create_gamma_lut()
    warm_sdr = np.zeros((1, 1, 3), dtype=np.float32)
    warm_u8 = np.zeros((1, 1, 3), dtype=np.uint8)
    postprocess_sdr_to_u8(warm_sdr, warm_u8, gamma_lut, 1, 1)

    if settings.DEVICE == "gpu":
        lut_path = os.path.join(project_root, "color-management", "acescg_to_srgb.npy")
        if not os.path.exists(lut_path):
            raise FileNotFoundError(
                f"ACES LUT not found: {lut_path}. Run color-management/generate-lut.py"
            )

        lut_host = np.load(lut_path).astype(np.float32)
        lut_device = cuda.to_device(lut_host)
        gamma_lut_device = cuda.to_device(gamma_lut)

        warm_fb = cuda.device_array((16, 16, 3), dtype=np.float32)
        warm_ldr = cuda.device_array((16, 16, 3), dtype=np.float32)
        warm_u8_device = cuda.device_array((16, 16, 3), dtype=np.uint8)
        tonemap_kernel[(1, 1), (16, 16)](
            warm_fb,
            warm_ldr,
            lut_device,
            np.int32(16),
            np.int32(16),
            np.float32(exposure_mul),
        )
        postprocess_full_gpu_kernel[(1, 1), (16, 16)](
            warm_fb,
            warm_u8_device,
            lut_device,
            gamma_lut_device,
            np.int32(16),
            np.int32(16),
            np.float32(exposure_mul),
        )
        cuda.synchronize()

    # Optional brute-force benchmark
    if settings.RENDER_NON_BVH_STATS:
        use_bvh = False
        t_brute = manager.run(grid, threads, locals())
        if settings.DEVICE == "gpu":
            cuda.synchronize()
        t = _phase_time("render (no ds)", t_brute)

        stats_brute = (
            out_stats.copy_to_host() if settings.DEVICE == "gpu" else out_stats
        )
        print_statistics(stats_brute, t - t_brute, len(triangles), is_ds=False)
        print()

        fb_hdr, out_stats = allocate_buffers(width_host, height_host)

    # Main render pass
    use_bvh = True
    if settings.DEVICE == "gpu":
        cuda.profile_start()
    t_bvh_start = manager.run(grid, threads, locals())
    if settings.DEVICE == "gpu":
        cuda.profile_stop()
        cuda.synchronize()

    t_bvh_end = _phase_time("render (with ds)", t_bvh_start)
    render_time = t_bvh_end - t_bvh_start

    # Post-processing and denoising
    if settings.DEVICE == "gpu":
        selected_tonemapper = str(settings.TONEMAPPER).lower()
        use_gpu_lut_path = selected_tonemapper in ("custom-aces", "lut", "acescg")
        can_denoise_gpu, can_denoise_cpu = get_denoise_path(use_gpu_lut_path)

        if not use_gpu_lut_path:
            # CPU tonemap path (custom, aces, etc.)
            print(
                f"[tonemapper] {selected_tonemapper} uses cpu tonemap path "
                "(gpu lut path is custom-aces-only for ACEScg->sRGB)"
            )

            if settings.DENOISE and can_denoise_gpu:
                print("[oidn] native cuda (hdr denoise on gpu, then cpu tonemap)")
                t_dn = time.perf_counter()
                denoise_gpu_hdr(fb_hdr, width_host, height_host)
                t = _phase_time("oidn denoise (cuda hdr)", t_dn)

            fb_hdr_host = fb_hdr.copy_to_host()
            t_tonemap = time.perf_counter()
            tonemap_hdr_to_sdr(
                fb_hdr_host, width_host, height_host, np.float32(exposure_mul)
            )
            t = _phase_time("tonemap HDR -> SDR", t_tonemap)

            if settings.DENOISE and can_denoise_cpu and not can_denoise_gpu:
                print("[oidn] pip cpu (cpu tonemap path, then denoise)")
                t_dn = time.perf_counter()
                denoise_cpu_ldr(fb_hdr_host, width_host, height_host)
                t = _phase_time("oidn denoise (pip cpu ldr)", t_dn)

            fb = np.zeros((height_host, width_host, 3), dtype=np.uint8)
            postprocess_sdr_to_u8(fb_hdr_host, fb, gamma_lut, width_host, height_host)
            t = _phase_time("gamma correct", t)
            render_time += t - t_bvh_end

        elif can_denoise_gpu or not settings.DENOISE:
            # Full GPU path with ACEScg LUT
            if settings.DENOISE and can_denoise_gpu:
                print("[oidn] native cuda (hdr denoise on gpu)")
                t_dn = time.perf_counter()
                denoise_gpu_hdr(fb_hdr, width_host, height_host)
                t = _phase_time("oidn denoise (cuda hdr)", t_dn)

            fb_u8_device = cuda.device_array(
                (height_host, width_host, 3), dtype=np.uint8
            )
            t_post = time.perf_counter()
            postprocess_full_gpu_kernel[grid, threads](
                fb_hdr,
                fb_u8_device,
                lut_device,
                gamma_lut_device,
                width,
                height,
                np.float32(exposure_mul),
            )
            cuda.synchronize()
            fb = fb_u8_device.copy_to_host()
            _phase_time("total time per frame", t_bvh_start, fps=True)

        else:
            # GPU tonemap + CPU denoise path
            print("[oidn] pip cpu (gpu tonemap, then denoise)")
            fb_ldr_device = cuda.device_array(
                (height_host, width_host, 3), dtype=np.float32
            )
            t_tonemap = time.perf_counter()
            tonemap_kernel[grid, threads](
                fb_hdr,
                fb_ldr_device,
                lut_device,
                width,
                height,
                np.float32(exposure_mul),
            )
            cuda.synchronize()
            fb_ldr_host = fb_ldr_device.copy_to_host()
            t = _phase_time("tonemap ", t_tonemap)

            if settings.DENOISE and can_denoise_cpu:
                t_dn = time.perf_counter()
                denoise_cpu_ldr(fb_ldr_host, width_host, height_host)
                t = _phase_time("oidn denoise (pip cpu ldr)", t_dn)

            fb = np.zeros((height_host, width_host, 3), dtype=np.uint8)
            postprocess_sdr_to_u8(fb_ldr_host, fb, gamma_lut, width_host, height_host)
            t = _phase_time("gamma correct", t)
            render_time += t - t_bvh_end

    else:
        # CPU render path
        fb_hdr_host = fb_hdr
        tonemap_hdr_to_sdr(
            fb_hdr_host, width_host, height_host, np.float32(exposure_mul)
        )
        t = _phase_time("tonemap HDR -> SDR", t)

        if settings.DENOISE:
            if HAS_PIP_OIDN:
                print("[oidn] pip cpu (tonemap first, then denoise)")
                t_dn = time.perf_counter()
                denoise_cpu_ldr(fb_hdr_host, width_host, height_host)
                t = _phase_time("oidn denoise (pip cpu ldr)", t_dn)
            elif HAS_NATIVE_CUDA_OIDN:
                warnings.warn(
                    "DENOISE is on and native cuda oidn is in utils/lib, but the "
                    "cpu renderer only supports pip oidn. Install the pip package "
                    "or set DEVICE gpu.",
                    RuntimeWarning,
                )
            else:
                warnings.warn(
                    "DENOISE is enabled but OIDN is not installed; skipping denoising.",
                    RuntimeWarning,
                )

        fb = np.zeros((height_host, width_host, 3), dtype=np.uint8)
        postprocess_sdr_to_u8(fb_hdr_host, fb, gamma_lut, width_host, height_host)
        t = _phase_time("gamma correct", t)
        render_time += t - t_bvh_end

    stats = out_stats.copy_to_host() if settings.DEVICE == "gpu" else out_stats
    print_statistics(stats, render_time, len(triangles))

    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "output", "output"
    )
    save_image(fb, output_path)

    print(f"\n[timing] {'total (+-)':<20}: {time.perf_counter() - t_start:7.2f} s")


__all__ = ["main"]
