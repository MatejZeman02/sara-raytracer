import math

import numba
import numpy as np
from numba import cuda, njit

from settings import DEVICE, MAX_BOUNCES
from constants import EPSILON, STACK_SIZE, ZERO, ONE, HALF

from utils import device_jit
from utils.vec_utils import (
    neg,
    vec3,
    add,
    cross,
    dot,
    mul,
    normalize,
    sub,
    linear_to_srgb,
)
from intersection import intersect_triangle, intersect_aabb
from shading import cook_torrance_shading


@device_jit
def render_normals(n, fb, x, y):
    """[Debug]: output normals as colors"""
    r_val = (n[0] + ONE) * HALF
    g_val = (n[1] + ONE) * HALF
    b_val = (n[2] + ONE) * HALF

    fb[y, x, 0] = min(255, int(r_val * np.float32(255.0)))
    fb[y, x, 1] = min(255, int(g_val * np.float32(255.0)))
    fb[y, x, 2] = min(255, int(b_val * np.float32(255.0)))


@device_jit
def is_valid_normal(n, ray_dir):
    """check if normal is valid (not facing wrong way)"""
    return dot(ray_dir, n) < ZERO


@device_jit
def get_tri_verts(triangles, idx):
    """fetch triangle vertices from array"""
    assert idx >= 0
    a = vec3(triangles[idx, 0, 0], triangles[idx, 0, 1], triangles[idx, 0, 2])
    b = vec3(triangles[idx, 1, 0], triangles[idx, 1, 1], triangles[idx, 1, 2])
    c = vec3(triangles[idx, 2, 0], triangles[idx, 2, 1], triangles[idx, 2, 2])
    return a, b, c


@device_jit
def get_closest_hit(triangles, bvh_nodes, use_bvh, ray_origin, ray_dir, inv_rd, stack):
    """traverse scene and return closest intersection data including barycentric coords"""
    closest_t = 1e20
    hit_idx = -1
    closest_u = ZERO
    closest_v = ZERO
    tri_tests = 0
    node_tests = 0

    if use_bvh:
        # stack is provided by the caller, reuse it
        stack_ptr = 0
        stack[stack_ptr] = 0

        while stack_ptr >= 0:
            assert stack_ptr < STACK_SIZE
            node_idx = stack[stack_ptr]
            stack_ptr -= 1
            node_tests += 1

            bmin = vec3(
                bvh_nodes[node_idx, 0], bvh_nodes[node_idx, 1], bvh_nodes[node_idx, 2]
            )
            bmax = vec3(
                bvh_nodes[node_idx, 3], bvh_nodes[node_idx, 4], bvh_nodes[node_idx, 5]
            )

            hit, tmin = intersect_aabb(ray_origin, inv_rd, bmin, bmax)

            # traverse children only if ray hits the box and is closer than current closest_t
            if hit and tmin < closest_t:
                data1 = bvh_nodes[node_idx, 6]  # positive for leaf nodes
                data2 = bvh_nodes[
                    node_idx, 7
                ]  # number of triangles for leaves or negative right child

                if data2 > ZERO:
                    start = int(data1)
                    count = int(data2)
                    for i in range(start, start + count):
                        tri_tests += 1
                        a, b, c = get_tri_verts(triangles, i)
                        t, u, v = intersect_triangle(ray_origin, ray_dir, a, b, c)
                        if EPSILON < t < closest_t:
                            closest_t = t
                            hit_idx = i
                            closest_u = u
                            closest_v = v
                else:
                    left_child = int(data1)
                    right_child = int(-data2)
                    stack_ptr += 1
                    stack[stack_ptr] = right_child
                    stack_ptr += 1
                    stack[stack_ptr] = left_child
    else:
        # good old brute-force intersection
        for i in range(triangles.shape[0]):
            tri_tests += 1
            a, b, c = get_tri_verts(triangles, i)
            t, u, v = intersect_triangle(ray_origin, ray_dir, a, b, c)
            if EPSILON < t < closest_t:
                closest_t = t
                hit_idx = i
                closest_u = u
                closest_v = v

    return closest_t, hit_idx, closest_u, closest_v, tri_tests, node_tests


@device_jit
def is_in_shadow(
    triangles, bvh_nodes, use_bvh, shadow_ro, d_l, inv_dl, dist_to_light, stack
):
    # any-hit traversal to determine if a point is shadowed
    assert dist_to_light > 0.0

    if not use_bvh:
        for i in range(triangles.shape[0]):
            ta, tb, tc = get_tri_verts(triangles, i)
            t, _u, _v = intersect_triangle(shadow_ro, d_l, ta, tb, tc)

            if EPSILON < t < dist_to_light:
                return True
        return False

    # stack is provided by the caller, reuse it
    stack_ptr = 0
    stack[stack_ptr] = 0

    while stack_ptr >= 0:
        assert stack_ptr < STACK_SIZE
        node_idx = stack[stack_ptr]
        stack_ptr -= 1

        bmin = vec3(
            bvh_nodes[node_idx, 0], bvh_nodes[node_idx, 1], bvh_nodes[node_idx, 2]
        )
        bmax = vec3(
            bvh_nodes[node_idx, 3], bvh_nodes[node_idx, 4], bvh_nodes[node_idx, 5]
        )

        hit, tmin = intersect_aabb(shadow_ro, inv_dl, bmin, bmax)

        # traverse only if box is hit and is closer than light source
        if hit and tmin < dist_to_light:
            data1 = bvh_nodes[node_idx, 6]
            data2 = bvh_nodes[node_idx, 7]

            if data2 > ZERO:
                start = int(data1)
                count = int(data2)
                for i in range(start, start + count):
                    ta, tb, tc = get_tri_verts(triangles, i)
                    t, _u, _v = intersect_triangle(shadow_ro, d_l, ta, tb, tc)

                    # immediate return on first valid blocking intersection
                    if EPSILON < t < dist_to_light:
                        return True
            else:
                left_child = int(data1)
                right_child = int(-data2)
                stack_ptr += 1
                stack[stack_ptr] = right_child
                stack_ptr += 1
                stack[stack_ptr] = left_child
    return False


@device_jit
def compute_inv_dir(dir_vec):
    """optimalization to divide only once per ray"""
    inv_x = ONE / dir_vec[0] if dir_vec[0] != ZERO else np.float32(1e15)
    inv_y = ONE / dir_vec[1] if dir_vec[1] != ZERO else np.float32(1e15)
    inv_z = ONE / dir_vec[2] if dir_vec[2] != ZERO else np.float32(1e15)
    return vec3(inv_x, inv_y, inv_z)


@device_jit
def compute_primary_ray(p00, qw, qh, origin, x, y):
    """primary ray generation"""
    xf = np.float32(x)
    yf = np.float32(y)
    dir_x = p00[0] + xf * qw[0] - yf * qh[0]
    dir_y = p00[1] + xf * qw[1] - yf * qh[1]
    dir_z = p00[2] + xf * qw[2] - yf * qh[2]

    ray_dir = normalize(vec3(dir_x, dir_y, dir_z))
    ray_origin = vec3(origin[0], origin[1], origin[2])
    inv_rd = compute_inv_dir(ray_dir)
    return ray_origin, ray_dir, inv_rd


@device_jit
def get_miss_color():
    """return dark gray background color"""
    val = np.float32(20.0) / np.float32(255.0)
    return vec3(val, val, val)


@device_jit
def get_emissive_color(materials, mat_idx):
    return vec3(materials[mat_idx, 7], materials[mat_idx, 8], materials[mat_idx, 9])


@device_jit
def get_vertex_normals(tri_normals, hit_idx):
    """fetch vertex normals for hit triangle from array"""
    na = vec3(
        tri_normals[hit_idx, 0, 0],
        tri_normals[hit_idx, 0, 1],
        tri_normals[hit_idx, 0, 2],
    )
    nb = vec3(
        tri_normals[hit_idx, 1, 0],
        tri_normals[hit_idx, 1, 1],
        tri_normals[hit_idx, 1, 2],
    )
    nc = vec3(
        tri_normals[hit_idx, 2, 0],
        tri_normals[hit_idx, 2, 1],
        tri_normals[hit_idx, 2, 2],
    )
    return na, nb, nc


@device_jit
def compute_surface_normal(triangles, tri_normals, hit_idx, ray_dir, hit_u, hit_v):
    """compute surface normal for hit triangle"""
    # fetch triangle vertices and vertex normals
    a, b, c = get_tri_verts(triangles, hit_idx)
    geom_n = normalize(cross(sub(b, a), sub(c, a)))
    is_backface = not is_valid_normal(geom_n, ray_dir)
    if is_backface:
        geom_n = neg(geom_n)

    na, nb, nc = get_vertex_normals(tri_normals, hit_idx)

    # throw error if vertex normals are missing from obj
    assert (
        na[0] != ZERO or na[1] != ZERO or na[2] != ZERO
    ), "vertex normals are missing from the obj file"
    # compute w barycentric weight
    w = ONE - hit_u - hit_v
    # verify coordinate sanity
    assert w >= -EPSILON and w <= ONE + EPSILON

    # interpolate normal using weights
    interp_x = w * na[0] + hit_u * nb[0] + hit_v * nc[0]
    interp_y = w * na[1] + hit_u * nb[1] + hit_v * nc[1]
    interp_z = w * na[2] + hit_u * nb[2] + hit_v * nc[2]

    n = normalize(vec3(interp_x, interp_y, interp_z))
    if is_backface:
        n = neg(n)

    return a, b, c, na, nb, nc, geom_n, n, w, is_backface


@device_jit
def compute_shadow_ray_origin(
    a, b, c, na, nb, nc, w, hit_u, hit_v, p, geom_n, is_backface
):
    """CHIANG'S analytical terminator offset @Disney 2019"""
    if na[0] == ZERO and na[1] == ZERO and na[2] == ZERO:
        return add(p, mul(geom_n, EPSILON))

    fa_ = neg(na) if is_backface else na
    fb_ = neg(nb) if is_backface else nb
    fc_ = neg(nc) if is_backface else nc

    p_a = add(p, mul(fa_, dot(sub(a, p), fa_)))
    p_b = add(p, mul(fb_, dot(sub(b, p), fb_)))
    p_c = add(p, mul(fc_, dot(sub(c, p), fc_)))

    p_curve = add(add(mul(p_a, w), mul(p_b, hit_u)), mul(p_c, hit_v))
    return add(p_curve, mul(geom_n, EPSILON))


@device_jit
def compute_shadowed(
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
):
    """compute if hit point is shadowed by tracing a ray towards the light source"""
    shadow_ro = compute_shadow_ray_origin(
        a, b, c, na, nb, nc, w, hit_u, hit_v, p, geom_n, is_backface
    )
    inv_dl = compute_inv_dir(d_l)
    return is_in_shadow(
        triangles, bvh_nodes, use_bvh, shadow_ro, d_l, inv_dl, dist_to_light, stack
    )


@device_jit
def compute_lit_color(materials, mat_idx, light_color, n, v_vec, d_l, shadowed):
    """Shading"""
    # fetch material properties
    r_d = vec3(materials[mat_idx, 0], materials[mat_idx, 1], materials[mat_idx, 2])
    r_s = vec3(materials[mat_idx, 3], materials[mat_idx, 4], materials[mat_idx, 5])
    h_val = materials[mat_idx, 6]
    l_color = vec3(light_color[0], light_color[1], light_color[2])

    if shadowed:
        return vec3(ZERO, ZERO, ZERO)
    return cook_torrance_shading(n, v_vec, d_l, r_d, r_s, h_val, l_color)


@device_jit
def write_color_to_fb(cr, cg, cb, fb, x, y):
    # apply srgb transfer function and write to framebuffer
    r_srgb = int(linear_to_srgb(cr) * 255)
    g_srgb = int(linear_to_srgb(cg) * 255)
    b_srgb = int(linear_to_srgb(cb) * 255)

    fb[y, x, 0] = min(255, r_srgb)
    fb[y, x, 1] = min(255, g_srgb)
    fb[y, x, 2] = min(255, b_srgb)


@device_jit
def compute_refraction(ray_dir, n, geom_n, p, ior, is_backface):
    """compute refraction ray direction and origin"""
    rel_ior = ior if is_backface else (ONE / ior if ior != ZERO else ONE)
    cos_i = dot(ray_dir, n)
    k = ONE - (rel_ior * rel_ior) * (ONE - (cos_i * cos_i))

    if k < ZERO:
        # total internal reflection
        new_dir = sub(ray_dir, mul(n, np.float32(2.0) * cos_i))
        new_origin = add(p, mul(geom_n, EPSILON))
    else:
        # pass through glass
        new_dir = normalize(
            add(
                mul(ray_dir, rel_ior),
                mul(n, rel_ior * (-cos_i) - np.float32(math.sqrt(k))),
            )
        )
        # step through the opposing geometry wall
        new_origin = sub(p, mul(geom_n, EPSILON))
    return new_dir, new_origin


@device_jit
def compute_reflection(ray_dir, n, geom_n, p):
    """compute reflection ray direction and origin"""
    cos_i = dot(ray_dir, n)
    new_dir = sub(ray_dir, mul(n, np.float32(2.0) * cos_i))
    new_origin = add(p, mul(geom_n, EPSILON))
    return new_dir, new_origin


# @device_jit(inline=False, device=False)
@device_jit()
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
    width,
    height,
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
        closest_t, hit_idx, hit_u, hit_v, tri_tests, node_tests = get_closest_hit(
            triangles, bvh_nodes, use_bvh, ray_origin, ray_dir, inv_rd, stack
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

        dist_to_light = np.float32(math.sqrt(dot(l_dir_vec, l_dir_vec)))
        assert dist_to_light > 0.0

        d_l = normalize(l_dir_vec)
        v_vec = normalize(sub(ray_origin, p))

        n_dot_l = dot(n, d_l)
        assert math.isfinite(n_dot_l)

        if n_dot_l <= 0.0:
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
                thr_r < np.float32(0.01)
                and thr_g < np.float32(0.01)
                and thr_b < np.float32(0.01)
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
        assert width > 0
        assert height > 0
        for y in numba.prange(height):
            for x in range(width):
                stack = np.empty(STACK_SIZE, dtype=np.int32)
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
                    width,
                    height,
                    x,
                    y,
                    stack,
                )

else:
    # CUDA GPU:
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
        stack = cuda.local.array(STACK_SIZE, dtype=numba.int32)
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
            width,
            height,
            x,
            y,
            stack,
        )
