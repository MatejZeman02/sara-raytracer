from math import isfinite
import numpy as np
from numpy import float32, empty
import numba
from numba import cuda, njit, prange
from settings import DEVICE, MAX_BOUNCES, SAMPLES
from constants import (
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
    MAT_TRANSMISSION_R,
    MAT_TRANSMISSION_G,
    MAT_TRANSMISSION_B,
    MAT_IOR,
    MAT_SPECULAR_R,
    MAT_SPECULAR_G,
    MAT_SPECULAR_B,
    MISS_COLOR_F,
)

from utils import device_jit
from utils.vec_utils import add, sub, mul, dot, normalize, vec3

from rays import (
    compute_primary_ray,
    compute_inv_dir,
    compute_refraction,
    compute_reflection,
)
from geometry import compute_surface_normal
from traversal import get_closest_hit
from materials import get_emissive_color, compute_shadowed, compute_lit_color
from framebuffer import write_hdr_to_fb
from lights import sample_area_light
from rng import rand_float32


@device_jit
def render_pixel(
    triangles,
    tri_normals,
    mat_indices,
    materials,
    bvh_nodes,
    use_bvh,
    p00,
    qw,
    qh,
    origin,
    fb_hdr,
    out_stats,
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
            closest_t, hit_idx, hit_u, hit_v, tri_tests, node_tests = get_closest_hit(
                triangles,
                bvh_nodes,
                use_bvh,
                ray_origin,
                ray_dir,
                inv_rd,
                stack,
                is_primary,
            )

            # accumulate statistics (last sample wins for the reset-at-primary logic):
            if is_primary:
                out_stats[y, x, PRIMARY_TRI] = tri_tests  # initial tri tests
                out_stats[y, x, PRIMARY_NODE] = node_tests  # initial node tests
                out_stats[y, x, PRIMARY_RAY] = 1  # exactly 1 primary ray
                out_stats[y, x, SECONDARY_RAY] = 0  # init secondary rays
                out_stats[y, x, SHADOW_RAY] = 0  # init shadow rays
            else:
                out_stats[y, x, PRIMARY_TRI] += tri_tests
                out_stats[y, x, PRIMARY_NODE] += node_tests
                out_stats[y, x, SECONDARY_RAY] += 1  # one more secondary ray

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
                # s_tri/s_node: shadow traversal test counts
                shadowed, s_tri, s_node = compute_shadowed(
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
                # accumulate traversal tests from shadow ray
                out_stats[y, x, PRIMARY_TRI] += s_tri
                out_stats[y, x, PRIMARY_NODE] += s_node

            # compute and add direct light to final color
            direct_color = compute_lit_color(
                materials, mat_idx, weighted_emission, n, v_vec, d_l, shadowed
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


if DEVICE == "cpu":
    # cpu entry point: parallel rows, serial columns
    @njit(parallel=False, fastmath=False)
    def render_kernel(
        triangles,
        tri_normals,
        mat_indices,
        materials,
        bvh_nodes,
        use_bvh,
        p00,
        qw,
        qh,
        origin,
        fb_hdr,
        out_stats,
        width,
        height,
        rng_states,
        emissive_tris,
        num_emissive,
    ):
        """cpu entry point, loops over all pixels with parallel rows"""
        assert width > 0
        assert height > 0

        DEBUG_PIXELS = [(440, 164), (434, 163), (250, 250)]
        for y in prange(height):
            for x in range(width):
                # if (x, y) not in DEBUG_PIXELS:
                #     continue
                # print("inspecting pixel x:", x, "y:", y)
                stack = empty(STACK_SIZE, dtype=np.int32)
                # set root
                stack[0] = np.int32(0)  # FIXME: not needed?
                thread_idx = y * width + x
                render_pixel(
                    triangles,
                    tri_normals,
                    mat_indices,
                    materials,
                    bvh_nodes,
                    use_bvh,
                    p00,
                    qw,
                    qh,
                    origin,
                    fb_hdr,
                    out_stats,
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
    @cuda.jit(fastmath=False)
    def render_kernel(
        triangles,
        tri_normals,
        mat_indices,
        materials,
        bvh_nodes,
        use_bvh,
        p00,
        qw,
        qh,
        origin,
        fb_hdr,
        out_stats,
        width,
        height,
        rng_states,
        emissive_tris,
        num_emissive,
    ):
        """gpu entry point, one thread per pixel"""
        x, y = cuda.grid(2)
        # return early for padding threads outside image bounds
        if x >= width or y >= height:
            return
        stack = cuda.local.array(STACK_SIZE, dtype=numba.int32)
        thread_idx = y * width + x
        render_pixel(
            triangles,
            tri_normals,
            mat_indices,
            materials,
            bvh_nodes,
            use_bvh,
            p00,
            qw,
            qh,
            origin,
            fb_hdr,
            out_stats,
            x,
            y,
            stack,
            rng_states,
            emissive_tris,
            num_emissive,
            thread_idx,
        )
