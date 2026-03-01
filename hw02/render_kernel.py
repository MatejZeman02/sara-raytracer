import math
import numba
from numba import cuda

from settings import EPSILON
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
    stack_size = 64
    closest_t = 1e20
    hit_idx = -1
    closest_u = 0.0
    closest_v = 0.0
    tri_tests = 0
    node_tests = 0

    if use_bvh:
        stack = cuda.local.array(stack_size, dtype=numba.int32)
        stack_ptr = 0
        stack[stack_ptr] = 0

        while stack_ptr >= 0:
            assert stack_ptr < stack_size

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
    stack_size = 64

    if not use_bvh:
        for i in range(triangles.shape[0]):
            ta, tb, tc = get_tri_verts(triangles, i)
            t, _u, _v = intersect_triangle(shadow_ro, d_l, ta, tb, tc)

            if EPSILON < t < dist_to_light:
                return True
        return False

    stack = cuda.local.array(stack_size, dtype=numba.int32)
    stack_ptr = 0
    stack[stack_ptr] = 0

    while stack_ptr >= 0:
        assert stack_ptr < stack_size

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


# @cuda.jit(debug=True, opt=False, cache=True)
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
    if x >= width or y >= height:
        return

    # primary ray generation
    dir_x = p00[0] + x * qw[0] - y * qh[0]
    dir_y = p00[1] + x * qw[1] - y * qh[1]
    dir_z = p00[2] + x * qw[2] - y * qh[2]

    ray_dir = normalize(vec3(dir_x, dir_y, dir_z))
    ray_origin = vec3(origin[0], origin[1], origin[2])

    # precompute inverse ray direction for faster aabb tests
    inv_dir_x = 1.0 / ray_dir[0] if ray_dir[0] != 0.0 else 1e15
    inv_dir_y = 1.0 / ray_dir[1] if ray_dir[1] != 0.0 else 1e15
    inv_dir_z = 1.0 / ray_dir[2] if ray_dir[2] != 0.0 else 1e15
    inv_rd = vec3(inv_dir_x, inv_dir_y, inv_dir_z)

    # find the closest triangle intersection
    closest_t, hit_idx, hit_u, hit_v, tri_tests, node_tests = get_closest_hit(
        triangles, bvh_nodes, use_bvh, ray_origin, ray_dir, inv_rd
    )

    out_stats[y, x, 0] = tri_tests
    out_stats[y, x, 1] = node_tests

    # no hit returns dark gray background
    if hit_idx == -1:
        fb[y, x, 0] = 20
        fb[y, x, 1] = 20
        fb[y, x, 2] = 20
        return

    # hit point material parameters
    mat_idx = mat_indices[hit_idx]
    ke_r = materials[mat_idx, 7]
    ke_g = materials[mat_idx, 8]
    ke_b = materials[mat_idx, 9]

    # pure emission without shading
    if ke_r > 0.0 or ke_g > 0.0 or ke_b > 0.0:
        fb[y, x, 0] = min(255, int(ke_r * 255))
        fb[y, x, 1] = min(255, int(ke_g * 255))
        fb[y, x, 2] = min(255, int(ke_b * 255))
        return

    # fetch hit triangle vertices
    a, b, c = get_tri_verts(triangles, hit_idx)

    geom_n = normalize(cross(sub(b, a), sub(c, a)))
    is_backface = not is_valid_normal(geom_n, ray_dir)  # if true normal facing wrong way
    if is_backface:
        geom_n = neg(geom_n)

    # fetch vertex normals
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

    # use fallback if vertex normals are missing from obj
    if na[0] == 0.0 and na[1] == 0.0 and na[2] == 0.0:
        assert (
            False
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

    # self shadowing check
    # if n_dot_l <= 0.0:
    #     shadowed = True
    # else:
    # offset origin along normal using geometric normal to prevent shadow acne
    shadow_ro = add(p, mul(geom_n, EPSILON))

    # precompute inverse shadow direction
    inv_dl_x = 1.0 / d_l[0] if d_l[0] != 0.0 else 1e15
    inv_dl_y = 1.0 / d_l[1] if d_l[1] != 0.0 else 1e15
    inv_dl_z = 1.0 / d_l[2] if d_l[2] != 0.0 else 1e15
    inv_dl = vec3(inv_dl_x, inv_dl_y, inv_dl_z)

    # test for shadows using any-hit optimization
    shadowed = is_in_shadow(
        triangles, bvh_nodes, use_bvh, shadow_ro, d_l, inv_dl, dist_to_light
    )

    # fetch material properties
    r_d = vec3(materials[mat_idx, 0], materials[mat_idx, 1], materials[mat_idx, 2])
    r_s = vec3(materials[mat_idx, 3], materials[mat_idx, 4], materials[mat_idx, 5])
    h_val = materials[mat_idx, 6]
    l_color = vec3(light_color[0], light_color[1], light_color[2])

    # apply shading based on visibility
    if shadowed:
        color = vec3(0.0, 0.0, 0.0)  # FIXME: for now black
    else:
        color = cook_torrance_shading(n, v_vec, d_l, r_d, r_s, h_val, l_color)

    # apply srgb transfer function and write to framebuffer
    r_srgb = linear_to_srgb(color[0])
    g_srgb = linear_to_srgb(color[1])
    b_srgb = linear_to_srgb(color[2])

    fb[y, x, 0] = min(255, int(r_srgb * 255))
    fb[y, x, 1] = min(255, int(g_srgb * 255))
    fb[y, x, 2] = min(255, int(b_srgb * 255))
