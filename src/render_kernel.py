from math import isfinite, ceil
import os
import numpy as np
from numpy import float32, empty
import numba
from numba import cuda, njit, prange
from .settings import DEVICE, MAX_BOUNCES, SAMPLES
from .constants import (
    STACK_SIZE,
    HALF,
    ZERO,
    ONE,
    THROUGHPUT_THRESHOLD,
    PRIMARY_TRI,
    PRIMARY_NODE,
    PRIMARY_RAY,
    SECONDARY_RAY,
    SHADOW_RAY,
    TRAVERSAL_DEPTH,
    TRAVERSE_TESTS,
    QUERY_DEPTH,
    MAT_TRANSMISSION_R,
    MAT_TRANSMISSION_G,
    MAT_TRANSMISSION_B,
    MAT_IOR,
    MAT_SPECULAR_R,
    MAT_SPECULAR_G,
    MAT_SPECULAR_B,
    MISS_COLOR_F,
    METRICS_NODE_TESTS,
    METRICS_TRI_TESTS,
    METRICS_SHADOW_TESTS,
    METRICS_IS_HIT,
)

from utils import device_jit
from utils.vec_utils import add, sub, mul, dot, normalize, vec3

from .rays import (
    compute_primary_ray,
    compute_inv_dir,
    compute_refraction,
    compute_reflection,
)
from .geometry import compute_surface_normal
from .traversal import get_closest_hit
from .materials import get_emissive_color, compute_shadowed, compute_lit_color
from .framebuffer import write_hdr_to_fb
from .lights import sample_area_light
from .rng import rand_float32


@device_jit
def render_pixel(
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
    use_bvh,
    p00,
    qw,
    qh,
    origin,
    fb_hdr,
    out_stats,
    metrics_out,
    x,
    y,
    stack,
    rng_states,
    emissive_tris,
    num_emissive,
    thread_idx,
):
    """per-pixel path tracing with multi-sample accumulation and sub-pixel jitter."""
    assert x >= 0
    assert y >= 0
    assert thread_idx >= 0
    assert num_emissive > 0

    f0 = ZERO
    f1 = ONE
    inv_samples = ONE / float32(SAMPLES)

    acc_r, acc_g, acc_b = f0, f0, f0

    # metrics accumulators for this pixel
    pixel_node_tests = 0
    pixel_tri_tests = 0
    pixel_shadow_tests = 0
    pixel_hit = 0

    for _ in range(SAMPLES):
        # sub-pixel jitter for free anti-aliasing: random offset in [-0.5, 0.5)
        jx = rand_float32(rng_states, thread_idx) - HALF  # jitter_x
        jy = rand_float32(rng_states, thread_idx) - HALF  # jitter_y

        ray_origin, ray_dir, inv_rd = compute_primary_ray(
            p00, qw, qh, origin, x, y, jx, jy
        )

        final_r, final_g, final_b = f0, f0, f0
        thr_r, thr_g, thr_b = f1, f1, f1

        for bounce in range(MAX_BOUNCES):
            is_primary = not bounce  # ray is primary, if 'bounce' is 0
            (
                closest_t,
                hit_idx,
                hit_u,
                hit_v,
                tri_tests,
                node_tests,
                traverse_tests,
                max_stack_depth,
            ) = get_closest_hit(
                triangles,
                bvh_nodes,
                use_bvh,
                ray_origin,
                ray_dir,
                inv_rd,
                stack,
                is_primary,
            )

            # accumulate gpu metrics
            pixel_node_tests += node_tests
            pixel_tri_tests += tri_tests

            if hit_idx >= 0:
                pixel_hit = 1

            # accumulate statistics (last sample wins for the reset-at-primary logic):
            if is_primary:
                out_stats[y, x, PRIMARY_TRI] = tri_tests  # initial tri tests
                out_stats[y, x, PRIMARY_NODE] = node_tests  # initial node tests
                out_stats[y, x, PRIMARY_RAY] = 1  # exactly 1 primary ray
                out_stats[y, x, SECONDARY_RAY] = 0  # init secondary rays
                out_stats[y, x, SHADOW_RAY] = 0  # init shadow rays
                out_stats[y, x, TRAVERSE_TESTS] = traverse_tests
                out_stats[y, x, QUERY_DEPTH] = max_stack_depth
            else:
                out_stats[y, x, PRIMARY_TRI] += tri_tests
                out_stats[y, x, PRIMARY_NODE] += node_tests
                out_stats[y, x, SECONDARY_RAY] += 1  # one more secondary ray
                out_stats[y, x, TRAVERSE_TESTS] += traverse_tests
                if max_stack_depth > out_stats[y, x, QUERY_DEPTH]:
                    out_stats[y, x, QUERY_DEPTH] = max_stack_depth

            if hit_idx == -1:
                # print("*")
                final_r += thr_r * MISS_COLOR_F[0]
                final_g += thr_g * MISS_COLOR_F[1]
                final_b += thr_b * MISS_COLOR_F[2]
                break

            mat_idx = mat_indices[hit_idx]
            # light emission
            em = get_emissive_color(materials, mat_idx)  # emissive_color
            if em[0] > f0 or em[1] > f0 or em[2] > f0:
                final_r += thr_r * em[0]
                final_g += thr_g * em[1]
                final_b += thr_b * em[2]
                break

            # a/b/c: triangle vertices; na/nb/nc: vertex normals; n: shading_normal; w: barycentric_w
            a, b, c, na, nb, nc, geom_n, n, w, is_backface = compute_surface_normal(
                triangles, tri_normals, hit_idx, ray_dir, hit_u, hit_v
            )

            p = add(ray_origin, mul(ray_dir, closest_t))  # hit_point
            v_vec = normalize(sub(ray_origin, p))  # view_direction

            # sample a random point on an emissive triangle (next-event estimation)
            d_l, dist_to_light, weighted_emission = (
                sample_area_light(  # d_l: direction_to_light
                    triangles,
                    mat_indices,
                    materials,
                    emissive_tris,
                    num_emissive,
                    p,
                    rng_states,
                    thread_idx,
                )
            )
            assert dist_to_light > ZERO

            n_dot_l = dot(n, d_l)  # normal_dot_light
            assert isfinite(n_dot_l)

            # skip shadow ray when the surface backfaces the sample or emission is zero
            no_emission = (
                weighted_emission[0] <= ZERO
                and weighted_emission[1] <= ZERO
                and weighted_emission[2] <= ZERO
            )
            if n_dot_l <= ZERO or no_emission:
                shadowed = True
            else:
                out_stats[y, x, SHADOW_RAY] += 1
                pixel_shadow_tests += 1
                # s_tri/s_node: shadow traversal test counts
                shadowed, s_tri, s_node, traverse_tests, max_stack_depth = (
                    compute_shadowed(
                        triangles,
                        bvh_nodes,
                        use_bvh,
                        a,
                        b,
                        c,
                        na,
                        nb,
                        nc,
                        w,
                        hit_u,
                        hit_v,
                        p,
                        geom_n,
                        is_backface,
                        d_l,
                        dist_to_light,
                        stack,
                    )
                )
                # accumulate traversal tests from shadow ray
                out_stats[y, x, PRIMARY_TRI] += s_tri
                out_stats[y, x, PRIMARY_NODE] += s_node

            # compute and add direct light to final color
            direct_color = compute_lit_color(
                materials,
                mat_idx,
                hit_idx,
                hit_u,
                hit_v,
                tri_uvs,
                mat_diffuse_tex_ids,
                diffuse_textures,
                tex_widths,
                tex_heights,
                weighted_emission,
                n,
                v_vec,
                d_l,
                shadowed,
            )
            final_r += thr_r * direct_color[0]
            final_g += thr_g * direct_color[1]
            final_b += thr_b * direct_color[2]
            # secondary ray routing
            tr_r = materials[mat_idx, MAT_TRANSMISSION_R]
            tr_g = materials[mat_idx, MAT_TRANSMISSION_G]
            tr_b = materials[mat_idx, MAT_TRANSMISSION_B]
            ior = materials[mat_idx, MAT_IOR]

            # specular
            ks_r = materials[mat_idx, MAT_SPECULAR_R]
            ks_g = materials[mat_idx, MAT_SPECULAR_G]
            ks_b = materials[mat_idx, MAT_SPECULAR_B]

            if tr_r > f0 or tr_g > f0 or tr_b > f0:
                ray_dir, ray_origin = compute_refraction(
                    ray_dir, n, geom_n, p, ior, is_backface
                )
                thr_r *= tr_r
                thr_g *= tr_g
                thr_b *= tr_b
                inv_rd = compute_inv_dir(ray_dir)
            elif ks_r > f0 or ks_g > f0 or ks_b > f0:
                # reflection
                ray_dir, ray_origin = compute_reflection(ray_dir, n, geom_n, p)
                thr_r *= ks_r
                thr_g *= ks_g
                thr_b *= ks_b
                inv_rd = compute_inv_dir(ray_dir)
                # stop tracing if throughput is negligible
                if (
                    thr_r < THROUGHPUT_THRESHOLD
                    and thr_g < THROUGHPUT_THRESHOLD
                    and thr_b < THROUGHPUT_THRESHOLD
                ):
                    break
            else:
                # diffuse objects stop light rays
                break

        acc_r += final_r
        acc_g += final_g
        acc_b += final_b

    # average accumulated samples and store raw HDR color
    write_hdr_to_fb(
        acc_r * inv_samples, acc_g * inv_samples, acc_b * inv_samples, fb_hdr, x, y
    )

    # write per-pixel bvh metrics
    if metrics_out is not None:
        idx = int(thread_idx)
        metrics_out[idx, METRICS_NODE_TESTS] = float32(pixel_node_tests)
        metrics_out[idx, METRICS_TRI_TESTS] = float32(pixel_tri_tests)
        metrics_out[idx, METRICS_SHADOW_TESTS] = float32(pixel_shadow_tests)
        metrics_out[idx, METRICS_IS_HIT] = float32(pixel_hit)


if DEVICE == "cpu":
    # cpu entry point: parallel rows, serial columns
    @njit(parallel=True, fastmath=True)
    def render_kernel(
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
        use_bvh,
        p00,
        qw,
        qh,
        origin,
        fb_hdr,
        out_stats,
        metrics_out,
        width,
        height,
        rng_states,
        emissive_tris,
        num_emissive,
    ):
        """cpu entry point, loops over all pixels with parallel rows"""
        assert width > 0
        assert height > 0

        for y in range(height):
            for x in range(width):
                stack = empty(STACK_SIZE, dtype=np.int32)
                # set root
                stack[0] = np.int32(0)  # FIXME: not needed?
                thread_idx = np.int32(y) * np.int32(width) + np.int32(x)
                render_pixel(
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
                    use_bvh,
                    p00,
                    qw,
                    qh,
                    origin,
                    fb_hdr,
                    out_stats,
                    metrics_out,
                    x,
                    y,
                    stack,
                    rng_states,
                    emissive_tris,
                    num_emissive,
                    thread_idx,
                )

elif DEVICE == "gpu":
    # gpu entry point: one cuda thread per pixel
    # @cuda.jit(fastmath=False, lineinfo=True) # lineinfo enables profiler line mapping
    @cuda.jit(fastmath=True)
    def render_kernel(
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
        use_bvh,
        p00,
        qw,
        qh,
        origin,
        fb_hdr,
        out_stats,
        metrics_out,
        width,
        height,
        rng_states,
        emissive_tris,
        num_emissive,
    ):
        """gpu entry point, one thread per pixel"""
        x, y = cuda.grid(2)
        x_i32 = numba.int32(x)
        y_i32 = numba.int32(y)
        width_i32 = numba.int32(width)
        height_i32 = numba.int32(height)
        # return early for padding threads outside image bounds
        if x_i32 >= width_i32 or y_i32 >= height_i32:
            return
        stack = cuda.local.array(STACK_SIZE, dtype=numba.int32)
        thread_idx = y_i32 * width_i32 + x_i32
        render_pixel(
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
            use_bvh,
            p00,
            qw,
            qh,
            origin,
            fb_hdr,
            out_stats,
            metrics_out,
            x_i32,
            y_i32,
            stack,
            rng_states,
            emissive_tris,
            num_emissive,
            thread_idx,
        )


def collect_bvh_stats(
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
    width,
    height,
    rng_states,
    emissive_tris,
    num_emissive,
    output_path,
):
    """collect gpu-side bvh traversal metrics and write to file."""
    assert width > 0
    assert height > 0
    assert DEVICE == "gpu"

    num_pixels = width * height

    # allocate metrics device array: (width*height, 4) with float32
    metrics_dev = cuda.device_array((num_pixels, 4), dtype=np.float32)

    # allocate dummy fb_hdr and out_stats (not used for metrics collection)
    fb_hdr_dummy = cuda.device_array((height, width, 3), dtype=np.float32)
    out_stats_dummy = cuda.device_array((height, width, 9), dtype=np.int32)

    threads = (16, 16)
    grid = (
        int(ceil(width / threads[0])),
        int(ceil(height / threads[1])),
    )

    # launch kernel with metrics collection
    render_kernel[grid, threads](
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
        True,  # use_bvh
        p00,
        qw,
        qh,
        origin,
        fb_hdr_dummy,
        out_stats_dummy,
        metrics_dev,
        np.int32(width),
        np.int32(height),
        rng_states,
        emissive_tris,
        num_emissive,
    )

    cuda.synchronize()

    # copy results back to host
    metrics_host = metrics_dev.copy_to_host()
    out_stats_host = out_stats_dummy.copy_to_host()

    # compute statistics
    node_tests = metrics_host[:, METRICS_NODE_TESTS]
    tri_tests = metrics_host[:, METRICS_TRI_TESTS]
    shadow_tests = metrics_host[:, METRICS_SHADOW_TESTS]
    is_hit = metrics_host[:, METRICS_IS_HIT]

    # extract traverse_tests and query_depth from out_stats (shape: height, width, 9)
    traverse_tests = out_stats_host[:, :, TRAVERSE_TESTS].ravel()
    query_depth = out_stats_host[:, :, QUERY_DEPTH].ravel()

    hit_mask = is_hit > 0
    hit_count = int(np.sum(hit_mask))
    miss_count = num_pixels - hit_count

    # compute stats for hit pixels only
    if hit_count > 0:
        node_tests_hit = node_tests[hit_mask]
        tri_tests_hit = tri_tests[hit_mask]
        shadow_tests_hit = shadow_tests[hit_mask]
    else:
        node_tests_hit = node_tests
        tri_tests_hit = tri_tests
        shadow_tests_hit = shadow_tests

    # build output lines
    lines = []
    lines.append("=" * 65)
    lines.append("  BVH METRICS COLLECTION (GPU-side)")
    lines.append("=" * 65)
    lines.append(f"Resolution:             {width} x {height} ({num_pixels:,} pixels)")
    lines.append(
        f"Hit pixels:             {hit_count:,} ({hit_count / num_pixels * 100:.1f}%)"
    )
    lines.append(
        f"Miss pixels (sky):      {miss_count:,} ({miss_count / num_pixels * 100:.1f}%)"
    )
    lines.append("-" * 65)
    lines.append(f"  {'Metric':<30} {'Min':>12} {'Max':>12} {'Mean':>12}")
    lines.append("-" * 65)

    def format_stats(name, values):
        lines.append(
            f"  {name:<28} {np.min(values):>12.1f} {np.max(values):>12.1f} {np.mean(values):>12.2f}"
        )

    format_stats("node_tests", node_tests)
    format_stats("tri_tests", tri_tests)
    format_stats("shadow_tests", shadow_tests)
    format_stats("traverse_tests", traverse_tests)
    format_stats("query_depth", query_depth)
    lines.append("-" * 65)
    lines.append("")

    if hit_count > 0:
        lines.append("  HIT PIXELS ONLY:")
        lines.append("-" * 65)
        lines.append(f"  {'Metric':<30} {'Min':>12} {'Max':>12} {'Mean':>12}")
        lines.append("-" * 65)
        format_stats("node_tests", node_tests_hit)
        format_stats("tri_tests", tri_tests_hit)
        format_stats("shadow_tests", shadow_tests_hit)
        format_stats(
            "traverse_tests",
            traverse_tests[hit_mask] if hit_count > 0 else traverse_tests,
        )
        format_stats(
            "query_depth", query_depth[hit_mask] if hit_count > 0 else query_depth
        )
        lines.append("-" * 65)
        lines.append("")

    lines.append(f"Total node_tests:         {np.sum(node_tests):,.0f}")
    lines.append(f"Total tri_tests:          {np.sum(tri_tests):,.0f}")
    lines.append(f"Total shadow_tests:       {np.sum(shadow_tests):,.0f}")
    lines.append(f"Total traverse_tests:     {np.sum(traverse_tests):,.0f}")
    lines.append("")
    lines.append(f"Overall mean node_tests:  {np.mean(node_tests):.2f}")
    lines.append(f"Overall mean tri_tests:   {np.mean(tri_tests):.2f}")
    lines.append(f"Overall mean shadow_tests:{np.mean(shadow_tests):.2f}")
    lines.append(f"Overall mean traverse_tests:{np.mean(traverse_tests):.2f}")
    lines.append(f"Overall mean query_depth: {np.mean(query_depth):.2f}")
    lines.append("")
    if hit_count > 0:
        lines.append(f"Hit-pixel mean node_tests:{np.mean(node_tests_hit):.2f}")
        lines.append(f"Hit-pixel mean tri_tests: {np.mean(tri_tests_hit):.2f}")
        lines.append(f"Hit-pixel mean shadow_tests:{np.mean(shadow_tests_hit):.2f}")
        ht = traverse_tests[hit_mask]
        hqd = query_depth[hit_mask]
        lines.append(f"Hit-pixel mean traverse_tests:{np.mean(ht):.2f}")
        lines.append(f"Hit-pixel mean query_depth: {np.mean(hqd):.2f}")
    lines.append("=" * 65)

    output_text = "\n".join(lines) + "\n"

    # ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w") as f:
        f.write(output_text)

    print(output_text)

    # --- construction metrics ---
    try:
        nodes = bvh_nodes.copy_to_host().reshape(-1, 8)
    except Exception:
        nodes = bvh_nodes

    if hasattr(nodes, "shape") and len(nodes.shape) == 2 and nodes.shape[1] >= 8:
        total = 0
        internal = 0
        leaves = 0
        leaf_depths = []
        leaf_prims = []
        stack = [0]
        while stack:
            idx = stack.pop()
            if idx < 0 or idx >= len(nodes):
                continue
            total += 1
            right_count = int(nodes[idx, 7])
            if right_count >= 0:
                # leaf
                leaves += 1
                leaf_prims.append(right_count)
                leaf_depths.append(len(stack))
            else:
                internal += 1
                left = int(nodes[idx, 6])
                right = -int(nodes[idx, 7])
                stack.append(left)
                stack.append(right)

        depth_min = min(leaf_depths) if leaf_depths else 0
        depth_max = max(leaf_depths) if leaf_depths else 0
        prims_min = min(leaf_prims) if leaf_prims else 0
        prims_max = max(leaf_prims) if leaf_prims else 0
        prims_mean = float(np.mean(leaf_prims)) if leaf_prims else 0.0
        mem_bytes = len(nodes) * nodes.dtype.itemsize * nodes.shape[1]
        mem_kb = mem_bytes / 1024.0

        lines.append("")
        lines.append("CONSTRUCTION METRICS:")
        lines.append("-" * 65)
        lines.append(f"  Total nodes:              {total:,}")
        lines.append(f"  Internal nodes:           {internal:,}")
        lines.append(f"  Leaf nodes:               {leaves:,}")
        lines.append(f"  Leaf depth (min/max):     {depth_min} / {depth_max}")
        lines.append(f"  Prims/leaf (min/max):     {prims_min} / {prims_max}")
        lines.append(f"  Prims/leaf (mean):        {prims_mean:.2f}")
        lines.append(
            f"  Memory:                   {mem_bytes:,} bytes ({mem_kb:.1f} KB)"
        )
        output_text = "\n".join(lines) + "\n"
        with open(output_path, "w") as f:
            f.write(output_text)
        print(output_text)

    return metrics_host
