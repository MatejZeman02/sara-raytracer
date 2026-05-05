"""Statistics collection and display."""

import math
import numpy as np
from .constants import (
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


def print_statistics(
    stats: np.ndarray, render_time: float, total_triangles: int, is_ds: bool = True
) -> None:
    """Calculate and print advanced rendering statistics."""
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

    # Calculate overall sums
    tot_prim = np.sum(prim_rays)
    tot_sec = np.sum(sec_rays)
    tot_shad = np.sum(shad_rays)
    tot_rays = tot_prim + tot_sec + tot_shad

    tot_tri = np.sum(tri_tests)
    tot_node = np.sum(node_tests)
    tot_inc = tot_tri + tot_node

    # Calculate performance
    mrays_sec = (tot_rays / 1e6) / render_time if render_time > 0 else 0

    # Calculate ratios and averages
    pct_prim = (tot_prim / tot_rays * 100) if tot_rays else 0
    pct_sec = (tot_sec / tot_rays * 100) if tot_rays else 0
    pct_shad = (tot_shad / tot_rays * 100) if tot_rays else 0

    node_tri_ratio = (tot_node / tot_tri) if tot_tri > 0 else 0
    tests_per_ray = (tot_inc / tot_rays) if tot_rays > 0 else 0
    nodes_per_ray = (tot_node / tot_rays) if tot_rays > 0 else 0

    optimal_log_nodes = math.log2(total_triangles)

    # Calculate per pixel arrays
    total_rays_per_px = prim_rays + sec_rays + shad_rays
    total_inc_per_px = tri_tests + node_tests

    # Calculate pixel workload distribution
    width = stats.shape[1]
    height = stats.shape[0]
    tot_px = width * height

    sky_mask = total_rays_per_px == 1
    sky_pixels = np.sum(sky_mask)

    hit_mask = total_rays_per_px > 1
    hit_incidences = total_inc_per_px[hit_mask]

    if len(hit_incidences) > 0:
        mean_inc = np.mean(hit_incidences)
        hard_thresh = mean_inc * 2.0
        hard_pixels = np.sum(hit_incidences > hard_thresh)
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
        f"  Avg query depth:      {np.mean(query_depth):.1f}\n"
        f"{SEP_DASH}\n"
    )

    if len(hit_incidences):
        print(
            f"PER_HIT-PIXEL LOAD (min / mean / max)\n"
            f"  Rays calls:        {np.min(total_rays_per_px[hit_mask])} / {np.mean(total_rays_per_px[hit_mask]):.1f} / {np.max(total_rays_per_px[hit_mask])}\n"
            f"  Incidence tests:   {np.min(total_inc_per_px[hit_mask])} / {np.mean(total_inc_per_px[hit_mask]):.1f} / {np.max(total_inc_per_px[hit_mask])}\n"
            f"{SEP_EQUAL}"
        )
