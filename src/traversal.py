from numpy import float32

from utils import device_jit
from utils.vec_utils import vec3
from intersection import intersect_triangle, intersect_aabb
from geometry import get_tri_verts
from constants import ZERO, EPSILON, STACK_SIZE


@device_jit
def get_closest_hit(triangles, bvh_nodes, use_bvh, ray_origin, ray_dir, inv_rd, stack):
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
