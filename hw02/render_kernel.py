import math
import numba
from numba import cuda

from settings import EPSILON, STACK_SIZE
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


@cuda.jit(device=True, inline=True, cache=True)
def render_normals(n, fb, x, y):
    """[Debug]: output normals as colors"""
    # map normal range [-1.0, 1.0] to rgb [0, 255]
    r_val = (n[0] + 1.0) * 0.5
    g_val = (n[1] + 1.0) * 0.5
    b_val = (n[2] + 1.0) * 0.5

    fb[y, x, 0] = min(255, int(r_val * 255))
    fb[y, x, 1] = min(255, int(g_val * 255))
    fb[y, x, 2] = min(255, int(b_val * 255))


@cuda.jit(device=True, inline=True, cache=True)
def is_valid_normal(n, ray_dir):
    """check if normal is valid (not facing wrong way)"""
    # simple check: normal should be facing opposite direction of ray
    # this is not a perfect test but can catch some cases of incorrect normals
    return dot(ray_dir, n) < 0.0


@cuda.jit(device=True, inline=True, cache=True)
def get_tri_verts(triangles, idx):
    # fetch triangle vertices from array
    assert idx >= 0

    a = vec3(triangles[idx, 0, 0], triangles[idx, 0, 1], triangles[idx, 0, 2])
    b = vec3(triangles[idx, 1, 0], triangles[idx, 1, 1], triangles[idx, 1, 2])
    c = vec3(triangles[idx, 2, 0], triangles[idx, 2, 1], triangles[idx, 2, 2])

    return a, b, c


@cuda.jit(device=True, inline=True, cache=True)
def get_closest_hit(triangles, bvh_nodes, use_bvh, ray_origin, ray_dir, inv_rd):
    # traverse scene and return closest intersection data including barycentric coords
    closest_t = 1e20
    hit_idx = -1
    closest_u = 0.0
    closest_v = 0.0
    tri_tests = 0
    node_tests = 0

    if use_bvh:
        stack = cuda.local.array(STACK_SIZE, dtype=numba.int32)
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

                if data2 > 0.0:
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
        # brute-force intersection
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


@cuda.jit(device=True, inline=True, cache=True)
def is_in_shadow(triangles, bvh_nodes, use_bvh, shadow_ro, d_l, inv_dl, dist_to_light):
    # any-hit traversal to determine if a point is shadowed
    assert dist_to_light > 0.0

    if not use_bvh:
        for i in range(triangles.shape[0]):
            ta, tb, tc = get_tri_verts(triangles, i)
            t, _u, _v = intersect_triangle(shadow_ro, d_l, ta, tb, tc)

            if EPSILON < t < dist_to_light:
                return True
        return False

    stack = cuda.local.array(STACK_SIZE, dtype=numba.int32)
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

            if data2 > 0.0:
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


@cuda.jit(device=True, inline=True, cache=True)
def compute_inv_dir(dir_vec):
    inv_x = 1.0 / dir_vec[0] if dir_vec[0] != 0.0 else 1e15
    inv_y = 1.0 / dir_vec[1] if dir_vec[1] != 0.0 else 1e15
    inv_z = 1.0 / dir_vec[2] if dir_vec[2] != 0.0 else 1e15
    return vec3(inv_x, inv_y, inv_z)


@cuda.jit(device=True, inline=True, cache=True)
def compute_primary_ray(p00, qw, qh, origin, x, y):
    # primary ray generation
    dir_x = p00[0] + x * qw[0] - y * qh[0]
    dir_y = p00[1] + x * qw[1] - y * qh[1]
    dir_z = p00[2] + x * qw[2] - y * qh[2]

    ray_dir = normalize(vec3(dir_x, dir_y, dir_z))
    ray_origin = vec3(origin[0], origin[1], origin[2])
    inv_rd = compute_inv_dir(ray_dir)
    return ray_origin, ray_dir, inv_rd


@cuda.jit(device=True, inline=True, cache=True)
def write_miss_color(fb, x, y):
    fb[y, x, 0] = 20.0
    fb[y, x, 1] = 20.0
    fb[y, x, 2] = 20.0


@cuda.jit(device=True, inline=True, cache=True)
def write_emissive_if_hit(materials, mat_idx, fb, x, y):
    ke_r = materials[mat_idx, 7]
    ke_g = materials[mat_idx, 8]
    ke_b = materials[mat_idx, 9]

    if ke_r > 0.0 or ke_g > 0.0 or ke_b > 0.0:
        fb[y, x, 0] = min(255, int(ke_r * 255))
        fb[y, x, 1] = min(255, int(ke_g * 255))
        fb[y, x, 2] = min(255, int(ke_b * 255))
        return True

    return False


@cuda.jit(device=True, inline=True, cache=True)
def get_vertex_normals(tri_normals, hit_idx):
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


@cuda.jit(device=True, inline=True, cache=True)
def compute_surface_normal(triangles, tri_normals, hit_idx, ray_dir, hit_u, hit_v):
    # fetch hit triangle vertices
    a, b, c = get_tri_verts(triangles, hit_idx)

    geom_n = normalize(cross(sub(b, a), sub(c, a)))
    is_backface = not is_valid_normal(
        geom_n, ray_dir
    )  # if true normal facing wrong way
    if is_backface:
        geom_n = neg(geom_n)

    na, nb, nc = get_vertex_normals(tri_normals, hit_idx)

    # throw error if vertex normals are missing from obj
    assert (
        na[0] != 0.0 or na[1] != 0.0 or na[2] != 0.0
    ), "\n\nAssertionError:\nVertex normals are missing from the obj file"

    # compute w barycentric weight
    w = 1.0 - hit_u - hit_v

    # verify coordinate sanity
    assert w >= -EPSILON and w <= 1.0 + EPSILON

    # interpolate normal using weights
    interp_x = w * na[0] + hit_u * nb[0] + hit_v * nc[0]
    interp_y = w * na[1] + hit_u * nb[1] + hit_v * nc[1]
    interp_z = w * na[2] + hit_u * nb[2] + hit_v * nc[2]

    n = normalize(vec3(interp_x, interp_y, interp_z))
    if is_backface:
        n = neg(n)

    return a, b, c, na, nb, nc, geom_n, n, w, is_backface


@cuda.jit(device=True, inline=True, cache=True)
def compute_shadow_ray_origin(
    a, b, c, na, nb, nc, w, hit_u, hit_v, p, geom_n, is_backface
):
    # CHIANG'S analytical terminator offset
    if na[0] == 0.0 and na[1] == 0.0 and na[2] == 0.0:
        # fallback for flat walls without vertex normals
        return add(p, mul(geom_n, EPSILON))

    # for curved objects with normals:
    # if we are inside, we must work with flipped normals
    fa_ = neg(na) if is_backface else na
    fb_ = neg(nb) if is_backface else nb
    fc_ = neg(nc) if is_backface else nc

    # project point P onto tangent planes of vertices
    t_a = dot(sub(a, p), fa_)
    p_a = add(p, mul(fa_, t_a))

    t_b = dot(sub(b, p), fb_)
    p_b = add(p, mul(fb_, t_b))

    t_c = dot(sub(c, p), fc_)
    p_c = add(p, mul(fc_, t_c))

    # barycentric interpolation of these projections (point on virtual curve)
    p_curve = add(add(mul(p_a, w), mul(p_b, hit_u)), mul(p_c, hit_v))

    # shift origin to this curve + add EPSILON safeguard against float precision errors
    return add(p_curve, mul(geom_n, EPSILON))


@cuda.jit(device=True, inline=True, cache=True)
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
):
    shadow_ro = compute_shadow_ray_origin(
        a, b, c, na, nb, nc, w, hit_u, hit_v, p, geom_n, is_backface
    )
    inv_dl = compute_inv_dir(d_l)
    return is_in_shadow(
        triangles, bvh_nodes, use_bvh, shadow_ro, d_l, inv_dl, dist_to_light
    )


@cuda.jit(device=True, inline=True, cache=True)
def compute_lit_color(materials, mat_idx, light_color, n, v_vec, d_l, shadowed):
    # fetch material properties
    r_d = vec3(materials[mat_idx, 0], materials[mat_idx, 1], materials[mat_idx, 2])
    r_s = vec3(materials[mat_idx, 3], materials[mat_idx, 4], materials[mat_idx, 5])
    h_val = materials[mat_idx, 6]
    l_color = vec3(light_color[0], light_color[1], light_color[2])

    # apply shading based on visibility
    if shadowed:
        return vec3(0.0, 0.0, 0.0)

    return cook_torrance_shading(n, v_vec, d_l, r_d, r_s, h_val, l_color)


@cuda.jit(device=True, inline=True, cache=True)
def write_color_to_fb(color, fb, x, y):
    # apply srgb transfer function and write to framebuffer
    r_srgb = int(linear_to_srgb(color[0]) * 255)
    g_srgb = int(linear_to_srgb(color[1]) * 255)
    b_srgb = int(linear_to_srgb(color[2]) * 255)

    fb[y, x, 0] = min(255, r_srgb)
    fb[y, x, 1] = min(255, g_srgb)
    fb[y, x, 2] = min(255, b_srgb)


################### MAIN RENDER KERNEL ###################


# @cuda.jit(debug=True, opt=False, cache=True) # turn on asserts
@cuda.jit(cache=True)
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
    x, y = cuda.grid(2)
    assert x < width and y < height

    ray_origin, ray_dir, inv_rd = compute_primary_ray(p00, qw, qh, origin, x, y)

    # find the closest triangle intersection
    closest_t, hit_idx, hit_u, hit_v, tri_tests, node_tests = get_closest_hit(
        triangles, bvh_nodes, use_bvh, ray_origin, ray_dir, inv_rd
    )

    out_stats[y, x, 0] = tri_tests
    out_stats[y, x, 1] = node_tests

    # no hit returns dark gray background
    if hit_idx == -1:
        write_miss_color(fb, x, y)
        return

    mat_idx = mat_indices[hit_idx]
    if write_emissive_if_hit(materials, mat_idx, fb, x, y):
        return

    a, b, c, na, nb, nc, geom_n, n, w, is_backface = compute_surface_normal(
        triangles, tri_normals, hit_idx, ray_dir, hit_u, hit_v
    )

    p = add(ray_origin, mul(ray_dir, closest_t))

    # light vectors
    l_pos = vec3(light_pos[0], light_pos[1], light_pos[2])
    l_dir_vec = sub(l_pos, p)

    # distance to light determines shadow ray limits
    dist_to_light = math.sqrt(dot(l_dir_vec, l_dir_vec))
    assert dist_to_light > 0.0

    d_l = normalize(l_dir_vec)
    v_vec = normalize(sub(ray_origin, p))

    n_dot_l = dot(n, d_l)

    # verify valid float computation
    assert math.isfinite(n_dot_l)

    # self shadowing check - optimization
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
        )

    color = compute_lit_color(materials, mat_idx, light_color, n, v_vec, d_l, shadowed)
    write_color_to_fb(color, fb, x, y)
