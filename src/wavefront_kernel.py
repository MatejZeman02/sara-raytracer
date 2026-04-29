from numpy import float32, int32
import numba

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
    WF_STATUS_DONE,
    WF_STATUS_ACTIVE,
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

render_kernel_pass1 = None
render_kernel_pass2 = None


@device_jit
def _load_vec3(buf, idx):
    return vec3(buf[idx, 0], buf[idx, 1], buf[idx, 2])


@device_jit
def _store_vec3(buf, idx, x, y, z):
    buf[idx, 0] = x
    buf[idx, 1] = y
    buf[idx, 2] = z


@device_jit
def _save_wavefront_state(
    state_ray_o,
    state_ray_d,
    state_accum,
    state_sample_accum,
    state_throughput,
    state_sample_idx,
    state_bounce_idx,
    state_in_path,
    state_mat_id,
    state_status,
    pixel_idx,
    ray_origin,
    ray_dir,
    acc_r,
    acc_g,
    acc_b,
    sample_r,
    sample_g,
    sample_b,
    thr_r,
    thr_g,
    thr_b,
    sample_idx,
    bounce,
    in_path,
    mat_id,
    status,
):
    _store_vec3(
        state_ray_o,
        pixel_idx,
        ray_origin[0],
        ray_origin[1],
        ray_origin[2],
    )
    _store_vec3(
        state_ray_d,
        pixel_idx,
        ray_dir[0],
        ray_dir[1],
        ray_dir[2],
    )
    _store_vec3(state_accum, pixel_idx, acc_r, acc_g, acc_b)
    _store_vec3(state_sample_accum, pixel_idx, sample_r, sample_g, sample_b)
    _store_vec3(state_throughput, pixel_idx, thr_r, thr_g, thr_b)
    state_sample_idx[pixel_idx] = sample_idx
    state_bounce_idx[pixel_idx] = bounce
    state_in_path[pixel_idx] = in_path
    state_mat_id[pixel_idx] = mat_id
    state_status[pixel_idx] = status


if DEVICE == "gpu":
    from numba import cuda

    @device_jit
    def _trace_wavefront_pixel(
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
        width,
        height,
        rng_states,
        emissive_tris,
        num_emissive,
        bvh_ops_budget,
        state_ray_o,
        state_ray_d,
        state_accum,
        state_sample_accum,
        state_throughput,
        state_sample_idx,
        state_bounce_idx,
        state_in_path,
        state_mat_id,
        state_status,
        pixel_idx,
        x,
        y,
        stack,
    ):
        if state_status[pixel_idx] == WF_STATUS_DONE:
            return

        inv_samples = ONE / float32(SAMPLES)
        budget_enabled = bvh_ops_budget > int32(0)
        ops_used = int32(0)

        ray_origin = _load_vec3(state_ray_o, pixel_idx)
        ray_dir = _load_vec3(state_ray_d, pixel_idx)

        acc = _load_vec3(state_accum, pixel_idx)
        acc_r = acc[0]
        acc_g = acc[1]
        acc_b = acc[2]

        sample_acc = _load_vec3(state_sample_accum, pixel_idx)
        sample_r = sample_acc[0]
        sample_g = sample_acc[1]
        sample_b = sample_acc[2]

        thr = _load_vec3(state_throughput, pixel_idx)
        thr_r = thr[0]
        thr_g = thr[1]
        thr_b = thr[2]

        sample_idx = state_sample_idx[pixel_idx]
        bounce = state_bounce_idx[pixel_idx]
        in_path = state_in_path[pixel_idx]
        last_mat_idx = int32(-1)

        while sample_idx < SAMPLES:
            if in_path == int32(0):
                # Begin a new sample path.
                jx = rand_float32(rng_states, pixel_idx) - HALF
                jy = rand_float32(rng_states, pixel_idx) - HALF
                ray_origin, ray_dir, _ = compute_primary_ray(
                    p00, qw, qh, origin, x, y, jx, jy
                )
                sample_r = ZERO
                sample_g = ZERO
                sample_b = ZERO
                thr_r = ONE
                thr_g = ONE
                thr_b = ONE
                bounce = int32(0)
                in_path = int32(1)
                last_mat_idx = int32(-1)

            while in_path == int32(1) and bounce < MAX_BOUNCES:
                is_primary = bounce == int32(0)
                inv_rd = compute_inv_dir(ray_dir)

                closest_t, hit_idx, hit_u, hit_v, tri_tests, node_tests = (
                    get_closest_hit(
                        triangles,
                        bvh_nodes,
                        use_bvh,
                        ray_origin,
                        ray_dir,
                        inv_rd,
                        stack,
                        is_primary,
                    )
                )

                ops_used += tri_tests + node_tests

                if is_primary:
                    out_stats[y, x, PRIMARY_TRI] = tri_tests
                    out_stats[y, x, PRIMARY_NODE] = node_tests
                    out_stats[y, x, PRIMARY_RAY] = 1
                    out_stats[y, x, SECONDARY_RAY] = 0
                    out_stats[y, x, SHADOW_RAY] = 0
                else:
                    out_stats[y, x, PRIMARY_TRI] += tri_tests
                    out_stats[y, x, PRIMARY_NODE] += node_tests
                    out_stats[y, x, SECONDARY_RAY] += 1

                if hit_idx == -1:
                    sample_r += thr_r * MISS_COLOR_F[0]
                    sample_g += thr_g * MISS_COLOR_F[1]
                    sample_b += thr_b * MISS_COLOR_F[2]
                    in_path = int32(0)
                    break

                mat_idx = mat_indices[hit_idx]
                last_mat_idx = mat_idx
                em = get_emissive_color(materials, mat_idx)
                if em[0] > ZERO or em[1] > ZERO or em[2] > ZERO:
                    sample_r += thr_r * em[0]
                    sample_g += thr_g * em[1]
                    sample_b += thr_b * em[2]
                    in_path = int32(0)
                    break

                a, b, c, na, nb, nc, geom_n, n, w, is_backface = compute_surface_normal(
                    triangles,
                    tri_normals,
                    hit_idx,
                    ray_dir,
                    hit_u,
                    hit_v,
                )

                p = add(ray_origin, mul(ray_dir, closest_t))
                v_vec = normalize(sub(ray_origin, p))

                d_l, dist_to_light, weighted_emission = sample_area_light(
                    triangles,
                    mat_indices,
                    materials,
                    emissive_tris,
                    num_emissive,
                    p,
                    rng_states,
                    pixel_idx,
                )

                n_dot_l = dot(n, d_l)
                no_emission = (
                    weighted_emission[0] <= ZERO
                    and weighted_emission[1] <= ZERO
                    and weighted_emission[2] <= ZERO
                )

                if n_dot_l <= ZERO or no_emission:
                    shadowed = True
                else:
                    out_stats[y, x, SHADOW_RAY] += 1
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
                    out_stats[y, x, PRIMARY_TRI] += s_tri
                    out_stats[y, x, PRIMARY_NODE] += s_node
                    ops_used += s_tri + s_node

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

                sample_r += thr_r * direct_color[0]
                sample_g += thr_g * direct_color[1]
                sample_b += thr_b * direct_color[2]

                tr_r = materials[mat_idx, MAT_TRANSMISSION_R]
                tr_g = materials[mat_idx, MAT_TRANSMISSION_G]
                tr_b = materials[mat_idx, MAT_TRANSMISSION_B]
                ior = materials[mat_idx, MAT_IOR]

                ks_r = materials[mat_idx, MAT_SPECULAR_R]
                ks_g = materials[mat_idx, MAT_SPECULAR_G]
                ks_b = materials[mat_idx, MAT_SPECULAR_B]

                if tr_r > ZERO or tr_g > ZERO or tr_b > ZERO:
                    ray_dir, ray_origin = compute_refraction(
                        ray_dir,
                        n,
                        geom_n,
                        p,
                        ior,
                        is_backface,
                    )
                    thr_r *= tr_r
                    thr_g *= tr_g
                    thr_b *= tr_b
                    bounce += 1
                elif ks_r > ZERO or ks_g > ZERO or ks_b > ZERO:
                    ray_dir, ray_origin = compute_reflection(ray_dir, n, geom_n, p)
                    thr_r *= ks_r
                    thr_g *= ks_g
                    thr_b *= ks_b
                    bounce += 1

                    if (
                        thr_r < THROUGHPUT_THRESHOLD
                        and thr_g < THROUGHPUT_THRESHOLD
                        and thr_b < THROUGHPUT_THRESHOLD
                    ):
                        in_path = int32(0)
                else:
                    in_path = int32(0)

                if (
                    budget_enabled
                    and ops_used >= bvh_ops_budget
                    and in_path == int32(1)
                ):
                    _save_wavefront_state(
                        state_ray_o,
                        state_ray_d,
                        state_accum,
                        state_sample_accum,
                        state_throughput,
                        state_sample_idx,
                        state_bounce_idx,
                        state_in_path,
                        state_mat_id,
                        state_status,
                        pixel_idx,
                        ray_origin,
                        ray_dir,
                        acc_r,
                        acc_g,
                        acc_b,
                        sample_r,
                        sample_g,
                        sample_b,
                        thr_r,
                        thr_g,
                        thr_b,
                        sample_idx,
                        bounce,
                        in_path,
                        last_mat_idx,
                        WF_STATUS_ACTIVE,
                    )
                    return

            if in_path == int32(1) and bounce >= MAX_BOUNCES:
                in_path = int32(0)

            if in_path == int32(0):
                acc_r += sample_r
                acc_g += sample_g
                acc_b += sample_b
                sample_r = ZERO
                sample_g = ZERO
                sample_b = ZERO
                bounce = int32(0)
                sample_idx += int32(1)
                thr_r = ONE
                thr_g = ONE
                thr_b = ONE
                last_mat_idx = int32(-1)

            if budget_enabled and ops_used >= bvh_ops_budget and sample_idx < SAMPLES:
                _save_wavefront_state(
                    state_ray_o,
                    state_ray_d,
                    state_accum,
                    state_sample_accum,
                    state_throughput,
                    state_sample_idx,
                    state_bounce_idx,
                    state_in_path,
                    state_mat_id,
                    state_status,
                    pixel_idx,
                    ray_origin,
                    ray_dir,
                    acc_r,
                    acc_g,
                    acc_b,
                    sample_r,
                    sample_g,
                    sample_b,
                    thr_r,
                    thr_g,
                    thr_b,
                    sample_idx,
                    bounce,
                    in_path,
                    last_mat_idx,
                    WF_STATUS_ACTIVE,
                )
                return

        write_hdr_to_fb(
            acc_r * inv_samples,
            acc_g * inv_samples,
            acc_b * inv_samples,
            fb_hdr,
            x,
            y,
        )

        _save_wavefront_state(
            state_ray_o,
            state_ray_d,
            state_accum,
            state_sample_accum,
            state_throughput,
            state_sample_idx,
            state_bounce_idx,
            state_in_path,
            state_mat_id,
            state_status,
            pixel_idx,
            ray_origin,
            ray_dir,
            acc_r,
            acc_g,
            acc_b,
            ZERO,
            ZERO,
            ZERO,
            ONE,
            ONE,
            ONE,
            sample_idx,
            int32(0),
            int32(0),
            int32(-1),
            WF_STATUS_DONE,
        )

    @cuda.jit(fastmath=True)
    def render_kernel_pass1(
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
        width,
        height,
        rng_states,
        emissive_tris,
        num_emissive,
        bvh_ops_budget,
        state_ray_o,
        state_ray_d,
        state_accum,
        state_sample_accum,
        state_throughput,
        state_sample_idx,
        state_bounce_idx,
        state_in_path,
        state_mat_id,
        state_status,
    ):
        x, y = cuda.grid(2)
        x_i32 = numba.int32(x)
        y_i32 = numba.int32(y)
        width_i32 = numba.int32(width)
        height_i32 = numba.int32(height)

        if x_i32 >= width_i32 or y_i32 >= height_i32:
            return

        pixel_idx = y_i32 * width_i32 + x_i32
        if state_status[pixel_idx] == WF_STATUS_DONE:
            return

        stack = cuda.local.array(STACK_SIZE, dtype=numba.int32)

        _trace_wavefront_pixel(
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
            width_i32,
            height_i32,
            rng_states,
            emissive_tris,
            num_emissive,
            bvh_ops_budget,
            state_ray_o,
            state_ray_d,
            state_accum,
            state_sample_accum,
            state_throughput,
            state_sample_idx,
            state_bounce_idx,
            state_in_path,
            state_mat_id,
            state_status,
            pixel_idx,
            x_i32,
            y_i32,
            stack,
        )

    @cuda.jit(fastmath=True)
    def render_kernel_pass2(
        active_indices,
        active_count,
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
        width,
        height,
        rng_states,
        emissive_tris,
        num_emissive,
        state_ray_o,
        state_ray_d,
        state_accum,
        state_sample_accum,
        state_throughput,
        state_sample_idx,
        state_bounce_idx,
        state_in_path,
        state_mat_id,
        state_status,
    ):
        tid = cuda.grid(1)
        if tid >= active_count:
            return

        pixel_idx = active_indices[tid]
        if state_status[pixel_idx] == WF_STATUS_DONE:
            return

        width_i32 = numba.int32(width)
        x_i32 = pixel_idx % width_i32
        y_i32 = pixel_idx // width_i32

        stack = cuda.local.array(STACK_SIZE, dtype=numba.int32)

        _trace_wavefront_pixel(
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
            width_i32,
            numba.int32(height),
            rng_states,
            emissive_tris,
            num_emissive,
            int32(0),  # budget disabled in cleanup pass
            state_ray_o,
            state_ray_d,
            state_accum,
            state_sample_accum,
            state_throughput,
            state_sample_idx,
            state_bounce_idx,
            state_in_path,
            state_mat_id,
            state_status,
            pixel_idx,
            x_i32,
            y_i32,
            stack,
        )
