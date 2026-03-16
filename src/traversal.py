from numpy import float32

from utils import device_jit
from utils.vec_utils import vec3
from intersection import intersect_triangle, intersect_aabb
from geometry import get_tri_verts
from constants import (
    ZERO,
    RAY_EPSILON,
    STACK_SIZE,
    BVH_MIN_X,
    BVH_MIN_Y,
    BVH_MIN_Z,
    BVH_MAX_X,
    BVH_MAX_Y,
    BVH_MAX_Z,
    BVH_LEFT_OR_START,
    BVH_RIGHT_OR_COUNT,
)


@device_jit
def get_closest_hit(
    triangles, bvh_nodes, use_bvh, ray_origin, ray_dir, inv_rd, stack, is_primary
):
    """traverse scene and return closest intersection data including barycentric coords"""
    closest_t = float32(1e20)
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
                bvh_nodes[node_idx, BVH_MIN_X],
                bvh_nodes[node_idx, BVH_MIN_Y],
                bvh_nodes[node_idx, BVH_MIN_Z],
            )
            bmax = vec3(
                bvh_nodes[node_idx, BVH_MAX_X],
                bvh_nodes[node_idx, BVH_MAX_Y],
                bvh_nodes[node_idx, BVH_MAX_Z],
            )

            hit, tmin = intersect_aabb(ray_origin, inv_rd, bmin, bmax)

            # traverse children only if ray hits the box and is closer than current closest_t
            if hit and tmin < closest_t:
                data1 = bvh_nodes[
                    node_idx, BVH_LEFT_OR_START
                ]  # positive for leaf nodes
                data2 = bvh_nodes[
                    node_idx, BVH_RIGHT_OR_COUNT
                ]  # number of triangles for leaves or negative right child

                if data2 > ZERO:
                    start = int(data1)
                    count = int(data2)
                    for i in range(start, start + count):
                        tri_tests += 1
                        a, b, c = get_tri_verts(triangles, i)
                        t, u, v = (
                            intersect_triangle(  # t: hit_distance; u/v: barycentric_uv
                                ray_origin, ray_dir, a, b, c, is_primary
                            )
                        )
                        if RAY_EPSILON < t < closest_t:
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
            t, u, v = intersect_triangle(
                ray_origin, ray_dir, a, b, c, is_primary
            )  # t: hit_distance; u/v: barycentric_uv
            if RAY_EPSILON < t < closest_t:
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
    tri_tests = 0
    node_tests = 0

    if not use_bvh:
        for i in range(triangles.shape[0]):
            tri_tests += 1
            ta, tb, tc = get_tri_verts(triangles, i)
            t, _u, _v = intersect_triangle(
                shadow_ro, d_l, ta, tb, tc, False
            )  # t: shadow_hit_distance

            if RAY_EPSILON < t < dist_to_light:
                return True, tri_tests, node_tests
        return False, tri_tests, node_tests

    # stack is provided by the caller, reuse it
    stack_ptr = 0
    stack[stack_ptr] = 0

    while stack_ptr >= 0:
        assert stack_ptr < STACK_SIZE
        node_idx = stack[stack_ptr]
        stack_ptr -= 1
        node_tests += 1

        bmin = vec3(
            bvh_nodes[node_idx, BVH_MIN_X],
            bvh_nodes[node_idx, BVH_MIN_Y],
            bvh_nodes[node_idx, BVH_MIN_Z],
        )
        bmax = vec3(
            bvh_nodes[node_idx, BVH_MAX_X],
            bvh_nodes[node_idx, BVH_MAX_Y],
            bvh_nodes[node_idx, BVH_MAX_Z],
        )

        hit, tmin = intersect_aabb(shadow_ro, inv_dl, bmin, bmax)

        # traverse only if box is hit and is closer than light source
        if hit and tmin < dist_to_light:
            data1 = bvh_nodes[node_idx, BVH_LEFT_OR_START]
            data2 = bvh_nodes[node_idx, BVH_RIGHT_OR_COUNT]

            if data2 > ZERO:
                start = int(data1)
                count = int(data2)
                for i in range(start, start + count):
                    tri_tests += 1
                    ta, tb, tc = get_tri_verts(triangles, i)
                    t, _u, _v = intersect_triangle(
                        shadow_ro, d_l, ta, tb, tc, False
                    )  # t: shadow_hit_distance

                    # immediate return on first valid blocking intersection
                    if RAY_EPSILON < t < dist_to_light:
                        return True, tri_tests, node_tests
            else:
                left_child = int(data1)
                right_child = int(-data2)
                stack_ptr += 1
                stack[stack_ptr] = right_child
                stack_ptr += 1
                stack[stack_ptr] = left_child

    return False, tri_tests, node_tests
