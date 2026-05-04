"""Homework renderer package entrypoint."""

# pylint: disable=too-many-locals, unused-variable, line-too-long

import time

LOAD_PYTHON_TIME = 1.3  # meassured using 'time' command on empty main().
t_start = time.perf_counter() - LOAD_PYTHON_TIME

import math
import os
import sys
import warnings

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
from numba.core.errors import NumbaPerformanceWarning
from PIL import Image

from .bvh import build_bvh
from .render_kernel import render_kernel, collect_bvh_stats
from .constants import (
    BLOCK_THREADS,
    SEED,
    MAT_EMISSIVE_R,
    MAT_EMISSIVE_G,
    MAT_EMISSIVE_B,
    PRIMARY_TRI,
    PRIMARY_NODE,
    PRIMARY_RAY,
    SECONDARY_RAY,
    SHADOW_RAY,
    TRAVERSAL_DEPTH,
    TRAVERSE_TESTS,
    QUERY_DEPTH,
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
from .denoiser import (
    HAS_NATIVE_CUDA_OIDN,
    HAS_PIP_OIDN,
    denoise_cuda_hdr,
    denoise_pip_ldr,
)

if settings.DEVICE == "gpu":
    from numba import cuda

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

project_root = _project_root

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
    if settings.USE_BVH_CACHE and os.path.exists(cache_file):
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
            triangles,
            tri_normals,
            tri_uvs,
            mat_indices,
            use_sah=settings.USE_SAH,
            use_binning=settings.USE_BINNING,
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
    # 5: max traversal depth (stack depth)
    # 6: query time (only for CPU)
    # 7: traverse tests
    # 8: query depth
    STATS_TUPLE = (height, width, 9)
    FB_HDR_SHAPE = (height, width, 3)  # float32 HDR framebuffer
    if settings.DEVICE == "gpu":
        fb_hdr = cuda.device_array(FB_HDR_SHAPE, dtype=np.float32)
        out_stats = cuda.device_array(STATS_TUPLE, dtype=np.int32)
    else:
        fb_hdr = np.zeros(FB_HDR_SHAPE, dtype=np.float32)
        out_stats = np.zeros(STATS_TUPLE, dtype=np.int32)

    return fb_hdr, out_stats


def print_statistics(stats, render_time, total_triangles, is_ds=True):
    """calculate and print advanced rendering statistics with a focus on readability and ratios."""
    if not settings.PRINT_STATS:
        return
    assert stats.ndim == 3
    assert stats.shape[2] == 9
    assert render_time >= 0.0
    assert total_triangles > 0

    tri_tests = stats[:, :, PRIMARY_TRI]
    node_tests = stats[:, :, PRIMARY_NODE]
    prim_rays = stats[:, :, PRIMARY_RAY]
    sec_rays = stats[:, :, SECONDARY_RAY]
    shad_rays = stats[:, :, SHADOW_RAY]
    max_stack_depth = stats[:, :, TRAVERSAL_DEPTH]
    traverse_tests = stats[:, :, TRAVERSE_TESTS]
    query_depth = stats[:, :, QUERY_DEPTH]

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
        f"  STATISTICS ({'DS' if is_ds else 'No DS'} on {settings.DEVICE.upper()})\n"
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
        f"TRAVERSAL STATS\n"
        f"  Max stack depth:      {np.max(max_stack_depth):d}\n"
        f"  Traverse tests:       {np.sum(traverse_tests):,}\n"
        f"  Avg query depth:  {np.mean(query_depth):.1f}\n"
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
    img.save(f"{output_path}.{settings.IMG_FORMAT}")
    print(f"Click to see the result onto: {output_path}.{settings.IMG_FORMAT}")


def main():
    """run the render pipeline."""
    # handle --collect-bvh-stats flag from command line
    collect_stats = settings.COLLECT_BVH_STATS
    for arg in sys.argv[1:]:
        if arg == "--collect-bvh-stats":
            collect_stats = True

    if settings.DEVICE == "gpu":
        print(f"Runs on device: {str(cuda.get_current_device().name)[1:]}")
    else:
        print(f"Runs on device: {settings.DEVICE.upper()}")
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
        t,
    ) = load_or_build_scene(json_file, cache_file, t)

    origin, p00, qw, qh, light_pos, light_color = build_setup_vectors(
        light_data, cam_data, width_host, height_host
    )
    fb_hdr, out_stats = allocate_buffers(width_host, height_host)
    if settings.DEVICE == "gpu":
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

    # dummy metrics array for kernel compatibility (not used when not collecting stats)
    metrics_out = cuda.device_array(
        (int(width_host) * int(height_host), 4), dtype=np.float32
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

    # if collect-bvh-stats is enabled, run metrics collection instead of full render
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

    # postprocess setup done before render timing so frame fps does not include one-time setup
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

        # warmup stage: pre-jit tiny postprocess kernels before timed render/post
        warm_fb = cuda.device_array((16, 16, 3), dtype=np.float32)
        warm_ldr = cuda.device_array((16, 16, 3), dtype=np.float32)
        warm_u8_device = cuda.device_array((16, 16, 3), dtype=np.uint8)
        tonemap_kernel[(1, 1), (16, 16)](
            warm_fb, warm_ldr, lut_device, np.int32(16), np.int32(16)
        )
        postprocess_full_gpu_kernel[(1, 1), (16, 16)](
            warm_fb,
            warm_u8_device,
            lut_device,
            gamma_lut_device,
            np.int32(16),
            np.int32(16),
        )
        cuda.synchronize()

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

        # reallocate buffers for the actual run
        fb_hdr, out_stats = allocate_buffers(width_host, height_host)

    use_bvh = True
    if settings.DEVICE == "gpu":
        cuda.profile_start()
    t_bvh_start = manager.run(grid, threads, locals())
    if settings.DEVICE == "gpu":
        cuda.profile_stop()
        cuda.synchronize()

    t_bvh_end = _phase_time("render (with ds)", t_bvh_start)
    render_time = t_bvh_end - t_bvh_start
    if settings.DEVICE == "gpu":
        selected_tonemapper = str(settings.TONEMAPPER).lower()
        # custom-aces (and legacy "lut"/"acescg") uses the baked ACEScg->sRGB LUT path.
        use_gpu_lut_path = selected_tonemapper in ("custom-aces", "lut", "acescg")
        use_native_full_gpu = HAS_NATIVE_CUDA_OIDN or not settings.DENOISE
        if settings.DENOISE and not HAS_NATIVE_CUDA_OIDN and HAS_PIP_OIDN:
            if use_gpu_lut_path:
                print("[oidn] pip cpu (aces lut tonemap first, then denoise)")
            else:
                print("[oidn] pip cpu (cpu tonemap path, then denoise)")
        elif settings.DENOISE and not HAS_NATIVE_CUDA_OIDN and not HAS_PIP_OIDN:
            warnings.warn(
                "DENOISE is enabled but OIDN is not installed; skipping denoising.",
                RuntimeWarning,
            )
            # no denoiser available, keep postprocessing on gpu
            use_native_full_gpu = True

        if not use_gpu_lut_path:
            print(
                f"[tonemapper] {selected_tonemapper} uses cpu tonemap path "
                "(gpu lut path is custom-aces-only for ACEScg->sRGB)"
            )

            if settings.DENOISE and HAS_NATIVE_CUDA_OIDN:
                print("[oidn] native cuda (hdr denoise on gpu, then cpu tonemap)")
                t_dn = time.perf_counter()
                denoise_cuda_hdr(fb_hdr, width_host, height_host)
                t = _phase_time("oidn denoise (cuda hdr)", t_dn)

            fb_hdr_host = fb_hdr.copy_to_host()
            t_tonemap = time.perf_counter()
            tonemap_hdr_to_sdr(fb_hdr_host, width_host, height_host)
            t = _phase_time("tonemap HDR -> SDR", t_tonemap)

            if settings.DENOISE and HAS_PIP_OIDN and not HAS_NATIVE_CUDA_OIDN:
                t_dn = time.perf_counter()
                denoise_pip_ldr(fb_hdr_host, width_host, height_host)
                t = _phase_time("oidn denoise (pip cpu ldr)", t_dn)

            fb = np.zeros((height_host, width_host, 3), dtype=np.uint8)
            postprocess_sdr_to_u8(fb_hdr_host, fb, gamma_lut, width_host, height_host)
            t = _phase_time("gamma correct", t)
            render_time += t - t_bvh_end
        elif use_native_full_gpu:
            if settings.DENOISE and HAS_NATIVE_CUDA_OIDN:
                print(
                    "[oidn] native cuda from utils/lib "
                    "(hdr denoise on gpu, then gpu tonemap+gamma)"
                )
                t_dn = time.perf_counter()
                denoise_cuda_hdr(fb_hdr, width_host, height_host)
                t = _phase_time("oidn denoise (cuda hdr)", t_dn)

            fb_u8_device = cuda.device_array((height_host, width_host, 3), dtype=np.uint8)
            t_post = time.perf_counter()
            postprocess_full_gpu_kernel[grid, threads](
                fb_hdr, fb_u8_device, lut_device, gamma_lut_device, width, height
            )
            cuda.synchronize()
            fb = fb_u8_device.copy_to_host()
            t = _phase_time("postprocess full gpu", t_post)
            render_time = t - t_bvh_start
            _phase_time("frame (render+post)", t_bvh_start, fps=True)
        else:
            fb_ldr_device = cuda.device_array((height_host, width_host, 3), dtype=np.float32)
            t_tonemap = time.perf_counter()
            tonemap_kernel[grid, threads](fb_hdr, fb_ldr_device, lut_device, width, height)
            cuda.synchronize()
            fb_ldr_host = fb_ldr_device.copy_to_host()
            t = _phase_time("tonemap ", t_tonemap)
            if settings.DENOISE and HAS_PIP_OIDN:
                t_dn = time.perf_counter()
                denoise_pip_ldr(fb_ldr_host, width_host, height_host)
                t = _phase_time("oidn denoise (pip cpu ldr)", t_dn)

            fb = np.zeros((height_host, width_host, 3), dtype=np.uint8)
            postprocess_sdr_to_u8(fb_ldr_host, fb, gamma_lut, width_host, height_host)
            t = _phase_time("gamma correct", t)
            render_time += t - t_bvh_end
    else:
        fb_hdr_host = fb_hdr
        tonemap_hdr_to_sdr(fb_hdr_host, width_host, height_host)
        t = _phase_time("tonemap HDR -> SDR", t)
        if settings.DENOISE:
            if HAS_PIP_OIDN:
                print("[oidn] pip cpu (tonemap first, then denoise)")
                t_dn = time.perf_counter()
                denoise_pip_ldr(fb_hdr_host, width_host, height_host)
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
        # add postprocess time to render time for more accurate "total time to final image" stat
        render_time += t - t_bvh_end

    stats = out_stats.copy_to_host() if settings.DEVICE == "gpu" else out_stats
    print_statistics(stats, render_time, len(triangles))

    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "output", "output"
    )
    save_image(fb, output_path)

    print(f"\n[timing] {'total (+-)':<20}: {time.perf_counter() - t_start:7.2f} s")


# expose 'main' as the only public symbol of the package
__all__ = ["main"]
