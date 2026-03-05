from math import sqrt, isfinite
from numpy import float32, int32, empty
from numba import cuda, njit, int32, prange

from settings import DEVICE, MAX_BOUNCES
from constants import STACK_SIZE, ZERO, ONE, THROUGHPUT_THRESHOLD

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
from framebuffer import get_miss_color, write_color_to_fb


@device_jit
def render_pixel(
    triangles,
    tri_normals,
    mat_indices,
    materials,
    bvh_nodes,
    use_bvh,
    light_pos,
    light_color,
    p00,
    qw,
    qh,
    origin,
    fb,
    out_stats,
    x,
    y,
    stack,
):
    """per-pixel ray tracing logic, called from cpu or gpu entry point"""
    assert x >= 0
    assert y >= 0

    ray_origin, ray_dir, inv_rd = compute_primary_ray(p00, qw, qh, origin, x, y)
    f0 = ZERO
    f1 = ONE

    final_r, final_g, final_b = f0, f0, f0
    thr_r, thr_g, thr_b = f1, f1, f1

    for bounce in range(MAX_BOUNCES):
        is_primary = not bounce
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

        # accumulate statistics
        if bounce == 0:
            out_stats[y, x, 0] = tri_tests
            out_stats[y, x, 1] = node_tests
        else:
            out_stats[y, x, 0] += tri_tests
            out_stats[y, x, 1] += node_tests

        if hit_idx == -1:
            mc = get_miss_color()
            final_r += thr_r * mc[0]
            final_g += thr_g * mc[1]
            final_b += thr_b * mc[2]
            break

        mat_idx = mat_indices[hit_idx]
        # light emission
        em = get_emissive_color(materials, mat_idx)
        if em[0] > f0 or em[1] > f0 or em[2] > f0:
            final_r += thr_r * em[0]
            final_g += thr_g * em[1]
            final_b += thr_b * em[2]
            break

        a, b, c, na, nb, nc, geom_n, n, w, is_backface = compute_surface_normal(
            triangles, tri_normals, hit_idx, ray_dir, hit_u, hit_v
        )

        p = add(ray_origin, mul(ray_dir, closest_t))

        l_pos = vec3(light_pos[0], light_pos[1], light_pos[2])
        l_dir_vec = sub(l_pos, p)

        dist_to_light = sqrt(dot(l_dir_vec, l_dir_vec))
        assert dist_to_light > 0.0

        d_l = normalize(l_dir_vec)
        v_vec = normalize(sub(ray_origin, p))

        n_dot_l = dot(n, d_l)
        assert isfinite(n_dot_l)

        if n_dot_l <= ZERO:
            shadowed = True
        else:
            shadowed = compute_shadowed(
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

        # compute and add direct light to final color
        direct_color = compute_lit_color(
            materials, mat_idx, light_color, n, v_vec, d_l, shadowed
        )
        final_r += thr_r * direct_color[0]
        final_g += thr_g * direct_color[1]
        final_b += thr_b * direct_color[2]
        # secondary ray routing
        tr_r = materials[mat_idx, 10]
        tr_g = materials[mat_idx, 11]
        tr_b = materials[mat_idx, 12]
        ior = materials[mat_idx, 13]

        ks_r = materials[mat_idx, 3]
        ks_g = materials[mat_idx, 4]
        ks_b = materials[mat_idx, 5]

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

    write_color_to_fb(final_r, final_g, final_b, fb, x, y)


if DEVICE == "cpu":
    # dual entry points based on device setting
    @njit(parallel=True, fastmath=True)
    def render_kernel(
        triangles,
        tri_normals,
        mat_indices,
        materials,
        bvh_nodes,
        use_bvh,
        light_pos,
        light_color,
        p00,
        qw,
        qh,
        origin,
        fb,
        out_stats,
        width,
        height,
    ):
        """cpu entry point, loops over all pixels with parallel rows"""
        assert width > ZERO
        assert height > ZERO
        for y in prange(height):
            for x in range(width):
                stack = empty(STACK_SIZE, dtype=int32)
                render_pixel(
                    triangles,
                    tri_normals,
                    mat_indices,
                    materials,
                    bvh_nodes,
                    use_bvh,
                    light_pos,
                    light_color,
                    p00,
                    qw,
                    qh,
                    origin,
                    fb,
                    out_stats,
                    x,
                    y,
                    stack,
                )

elif DEVICE == "gpu":
    # CUDA GPU:
    # @cuda.jit(fastmath=True, lineinfo=True) # info in compiled code
    @cuda.jit(fastmath=True)
    def render_kernel(
        triangles,
        tri_normals,
        mat_indices,
        materials,
        bvh_nodes,
        use_bvh,
        light_pos,
        light_color,
        p00,
        qw,
        qh,
        origin,
        fb,
        out_stats,
        width,
        height,
    ):
        """gpu entry point, one thread per pixel"""
        x, y = cuda.grid(2)
        # return early for padding threads (outside image)
        if x >= width or y >= height:
            return
        stack = cuda.local.array(STACK_SIZE, dtype=int32)
        render_pixel(
            triangles,
            tri_normals,
            mat_indices,
            materials,
            bvh_nodes,
            use_bvh,
            light_pos,
            light_color,
            p00,
            qw,
            qh,
            origin,
            fb,
            out_stats,
            x,
            y,
            stack,
        )
