"""CUDA render kernel for hw01."""

import numba  # type: ignore
from numba import cuda  # type: ignore

from utils.vec_utils import vec3, add, cross, dot, mul, normalize, sub, linear_to_srgb
from intersection import intersect_triangle, intersect_aabb
from shading import cook_torrance_shading, phong_shading


@cuda.jit
def render_kernel(
    triangles,
    mat_indices,
    materials,
    bvh_nodes,  # array of bvh nodes
    use_bvh,  # boolean flag
    light_pos,
    light_color,
    p00,
    qw,
    qh,
    origin,
    fb,
    out_stats,  # array for statistics
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

    # precomputing inverse ray direction for faster aabb tests
    inv_dir_x = 1.0 / ray_dir[0] if ray_dir[0] != 0.0 else 1e15
    inv_dir_y = 1.0 / ray_dir[1] if ray_dir[1] != 0.0 else 1e15
    inv_dir_z = 1.0 / ray_dir[2] if ray_dir[2] != 0.0 else 1e15
    inv_rd = vec3(inv_dir_x, inv_dir_y, inv_dir_z)

    closest_t = 1e20
    hit_idx = -1

    tri_tests = 0
    node_tests = 0

    if use_bvh:
        # local stack for iterative bvh traversal
        MAX_STACK_SIZE = 64
        stack = cuda.local.array(MAX_STACK_SIZE, dtype=numba.int32)
        stack_ptr = 0
        stack[stack_ptr] = 0

        while stack_ptr >= 0:
            assert stack_ptr < MAX_STACK_SIZE

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
                data1 = bvh_nodes[node_idx, 6]
                data2 = bvh_nodes[node_idx, 7]

                if data2 > 0.0:
                    # leaf node, intersect with triangles
                    start = int(data1)
                    count = int(data2)
                    for i in range(start, start + count):
                        tri_tests += 1

                        a = vec3(
                            triangles[i, 0, 0], triangles[i, 0, 1], triangles[i, 0, 2]
                        )
                        b = vec3(
                            triangles[i, 1, 0], triangles[i, 1, 1], triangles[i, 1, 2]
                        )
                        c = vec3(
                            triangles[i, 2, 0], triangles[i, 2, 1], triangles[i, 2, 2]
                        )

                        t, _u, _v = intersect_triangle(ray_origin, ray_dir, a, b, c)
                        if 0.001 < t < closest_t:
                            closest_t = t
                            hit_idx = i
                else:
                    # inner node, push children to stack
                    left_child = int(data1)
                    right_child = int(-data2)

                    stack_ptr += 1
                    stack[stack_ptr] = right_child
                    stack_ptr += 1
                    stack[stack_ptr] = left_child
    else:
        # brute force fallback
        for i in range(triangles.shape[0]):
            tri_tests += 1

            a = vec3(triangles[i, 0, 0], triangles[i, 0, 1], triangles[i, 0, 2])
            b = vec3(triangles[i, 1, 0], triangles[i, 1, 1], triangles[i, 1, 2])
            c = vec3(triangles[i, 2, 0], triangles[i, 2, 1], triangles[i, 2, 2])

            t, _u, _v = intersect_triangle(ray_origin, ray_dir, a, b, c)
            if 0.001 < t < closest_t:
                closest_t = t
                hit_idx = i

    # save stats for this pixel
    out_stats[y, x, 0] = tri_tests
    out_stats[y, x, 1] = node_tests

    # shading logic remains the same
    if hit_idx == -1:
        fb[y, x, 0] = 20
        fb[y, x, 1] = 20
        fb[y, x, 2] = 20
        return

    mat_idx = mat_indices[hit_idx]
    ke_r = materials[mat_idx, 7]
    ke_g = materials[mat_idx, 8]
    ke_b = materials[mat_idx, 9]

    if ke_r > 0.0 or ke_g > 0.0 or ke_b > 0.0:
        fb[y, x, 0] = min(255, int(ke_r * 255))
        fb[y, x, 1] = min(255, int(ke_g * 255))
        fb[y, x, 2] = min(255, int(ke_b * 255))
    else:
        a = vec3(
            triangles[hit_idx, 0, 0], triangles[hit_idx, 0, 1], triangles[hit_idx, 0, 2]
        )
        b = vec3(
            triangles[hit_idx, 1, 0], triangles[hit_idx, 1, 1], triangles[hit_idx, 1, 2]
        )
        c = vec3(
            triangles[hit_idx, 2, 0], triangles[hit_idx, 2, 1], triangles[hit_idx, 2, 2]
        )

        n = normalize(cross(sub(b, a), sub(c, a)))
        p = add(ray_origin, mul(ray_dir, closest_t))

        l_pos = vec3(light_pos[0], light_pos[1], light_pos[2])
        d_l = normalize(sub(l_pos, p))
        v_vec = normalize(sub(ray_origin, p))

        r_d = vec3(materials[mat_idx, 0], materials[mat_idx, 1], materials[mat_idx, 2])
        r_s = vec3(materials[mat_idx, 3], materials[mat_idx, 4], materials[mat_idx, 5])
        h_val = materials[mat_idx, 6]
        l_color = vec3(light_color[0], light_color[1], light_color[2])

        color = cook_torrance_shading(n, v_vec, d_l, r_d, r_s, h_val, l_color)

        r_srgb = linear_to_srgb(color[0])
        g_srgb = linear_to_srgb(color[1])
        b_srgb = linear_to_srgb(color[2])

        fb[y, x, 0] = min(255, int(r_srgb * 255))
        fb[y, x, 1] = min(255, int(g_srgb * 255))
        fb[y, x, 2] = min(255, int(b_srgb * 255))
