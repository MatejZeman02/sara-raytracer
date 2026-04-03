"""SAH (surface area heuristic) BVH builder for ray tracing on cpu with jit using stack"""

import numpy as np
from numba import njit
from constants import (
    BVH_MIN_X,
    BVH_MIN_Y,
    BVH_MIN_Z,
    BVH_MAX_X,
    BVH_MAX_Y,
    BVH_MAX_Z,
    BVH_LEFT_OR_START,
    BVH_RIGHT_OR_COUNT,
)

BINS = 8


@njit
def get_aabb(triangles, tri_ids, start, end):
    """Bounding box for a subset of triangles"""
    assert start < end

    bmin = np.array([1e20, 1e20, 1e20], dtype=np.float32)
    bmax = np.array([-1e20, -1e20, -1e20], dtype=np.float32)

    for i in range(start, end):
        t_idx = tri_ids[i]

        # FIXME: loop over the 3 vertices, unrolling the 3 axes (x, y, z)
        for j in range(3):
            v_x = triangles[t_idx, j, 0]
            v_y = triangles[t_idx, j, 1]
            v_z = triangles[t_idx, j, 2]

            if v_x < bmin[0]:
                bmin[0] = v_x
            if v_x > bmax[0]:
                bmax[0] = v_x

            if v_y < bmin[1]:
                bmin[1] = v_y
            if v_y > bmax[1]:
                bmax[1] = v_y

            if v_z < bmin[2]:
                bmin[2] = v_z
            if v_z > bmax[2]:
                bmax[2] = v_z
    return bmin, bmax


@njit
def get_area(bmin, bmax):
    """calculate surface area of bounding box"""
    e = bmax - bmin
    if e[0] < 0.0 or e[1] < 0.0 or e[2] < 0.0:
        return 0.0
    return 2.0 * (e[0] * e[1] + e[1] * e[2] + e[2] * e[0])


@njit
def build_bvh_jit(triangles, centroids, tri_ids, nodes):
    """
    Iterative sah bvh builder using stack.
    It looks scary, it has 6x nested loops.
    I know. But it shouldn't be so bad...
    """
    assert len(triangles) > 0

    stack = np.zeros((8192, 3), dtype=np.int32)
    stack_ptr = 1
    # packing may not work with numba
    stack[0, 0] = 0
    stack[0, 1] = 0
    stack[0, 2] = len(tri_ids)

    nodes_used = 1

    while stack_ptr > 0:
        stack_ptr -= 1
        # unpacking may not work with numba
        node_idx = stack[stack_ptr, 0]
        start = stack[stack_ptr, 1]
        end = stack[stack_ptr, 2]

        num_tris = end - start
        bmin, bmax = get_aabb(triangles, tri_ids, start, end)

        # force leaf if too few triangles
        if num_tris <= 2:
            nodes[node_idx, BVH_MIN_X : BVH_MIN_Z + 1] = bmin
            nodes[node_idx, BVH_MAX_X : BVH_MAX_Z + 1] = bmax
            nodes[node_idx, BVH_LEFT_OR_START] = start
            nodes[node_idx, BVH_RIGHT_OR_COUNT] = num_tris
            continue

        # centroid bounds for binning
        cmin = np.array([1e20, 1e20, 1e20], dtype=np.float32)
        cmax = np.array([-1e20, -1e20, -1e20], dtype=np.float32)
        for i in range(start, end):
            t_idx = tri_ids[i]
            # unrolled 'for k in range(3)'
            val_x = centroids[t_idx, 0]
            val_y = centroids[t_idx, 1]
            val_z = centroids[t_idx, 2]

            if val_x < cmin[0]:
                cmin[0] = val_x
            if val_x > cmax[0]:
                cmax[0] = val_x

            if val_y < cmin[1]:
                cmin[1] = val_y
            if val_y > cmax[1]:
                cmax[1] = val_y

            if val_z < cmin[2]:
                cmin[2] = val_z
            if val_z > cmax[2]:
                cmax[2] = val_z

        best_cost = 1e20
        best_axis = -1
        best_split = 0.0

        # test sah over bins per axis
        for axis in range(3):
            if cmin[axis] == cmax[axis]:
                continue

            scale = BINS / (cmax[axis] - cmin[axis])
            for b in range(1, BINS):
                split_val = cmin[axis] + b / scale

                nl = 0
                nr = 0
                lmin = np.array([1e20, 1e20, 1e20], dtype=np.float32)
                lmax = np.array([-1e20, -1e20, -1e20], dtype=np.float32)
                rmin = np.array([1e20, 1e20, 1e20], dtype=np.float32)
                rmax = np.array([-1e20, -1e20, -1e20], dtype=np.float32)

                # accumulate bounding boxes for left and right sides
                for i in range(start, end):
                    t_idx = tri_ids[i]
                    if centroids[t_idx, axis] < split_val:
                        nl += 1
                        for j in range(3):
                            # unrolled 'for k in range(3)'
                            val_x = triangles[t_idx, j, 0]
                            val_y = triangles[t_idx, j, 1]
                            val_z = triangles[t_idx, j, 2]

                            if val_x < lmin[0]:
                                lmin[0] = val_x
                            if val_x > lmax[0]:
                                lmax[0] = val_x

                            if val_y < lmin[1]:
                                lmin[1] = val_y
                            if val_y > lmax[1]:
                                lmax[1] = val_y

                            if val_z < lmin[2]:
                                lmin[2] = val_z
                            if val_z > lmax[2]:
                                lmax[2] = val_z
                    else:
                        nr += 1
                        for j in range(3):
                            for k in range(3):
                                val = triangles[t_idx, j, k]
                                if val < rmin[k]:
                                    rmin[k] = val
                                if val > rmax[k]:
                                    rmax[k] = val

                if nl == 0 or nr == 0:
                    continue

                area_l = get_area(lmin, lmax)
                area_r = get_area(rmin, rmax)
                area_p = get_area(bmin, bmax)

                cost = 1.0 + (nl * area_l + nr * area_r) / area_p
                if cost < best_cost:
                    best_cost = cost
                    best_axis = axis
                    best_split = split_val

        # make leaf if sah cost is worse than just intersecting all
        if best_cost >= num_tris:
            nodes[node_idx, BVH_MIN_X : BVH_MIN_Z + 1] = bmin
            nodes[node_idx, BVH_MAX_X : BVH_MAX_Z + 1] = bmax
            nodes[node_idx, BVH_LEFT_OR_START] = start
            nodes[node_idx, BVH_RIGHT_OR_COUNT] = num_tris
            continue

        # partition triangle indices in place
        left_idx = start
        right_idx = end - 1
        while left_idx <= right_idx:
            if centroids[tri_ids[left_idx], best_axis] < best_split:
                left_idx += 1
            else:
                tmp = tri_ids[left_idx]
                tri_ids[left_idx] = tri_ids[right_idx]
                tri_ids[right_idx] = tmp
                right_idx -= 1

        split_idx = left_idx

        # if sah fails to split, just make a leaf
        if split_idx == start or split_idx == end:
            nodes[node_idx, BVH_MIN_X : BVH_MIN_Z + 1] = bmin
            nodes[node_idx, BVH_MAX_X : BVH_MAX_Z + 1] = bmax
            nodes[node_idx, BVH_LEFT_OR_START] = start
            nodes[node_idx, BVH_RIGHT_OR_COUNT] = num_tris
            continue

        # allocate children
        left_child = nodes_used
        right_child = nodes_used + 1
        nodes_used += 2
        nodes[node_idx, BVH_MIN_X : BVH_MIN_Z + 1] = bmin
        nodes[node_idx, BVH_MAX_X : BVH_MAX_Z + 1] = bmax
        nodes[node_idx, BVH_LEFT_OR_START] = left_child
        nodes[node_idx, BVH_RIGHT_OR_COUNT] = -right_child

        # push to stack larger first to minimize stack depth
        if split_idx - start > end - split_idx:
            stack[stack_ptr, 0] = left_child
            stack[stack_ptr, 1] = start
            stack[stack_ptr, 2] = split_idx
            stack_ptr += 1

            stack[stack_ptr, 0] = right_child
            stack[stack_ptr, 1] = split_idx
            stack[stack_ptr, 2] = end
            stack_ptr += 1
        else:
            stack[stack_ptr, 0] = right_child
            stack[stack_ptr, 1] = split_idx
            stack[stack_ptr, 2] = end
            stack_ptr += 1

            stack[stack_ptr, 0] = left_child
            stack[stack_ptr, 1] = start
            stack[stack_ptr, 2] = split_idx
            stack_ptr += 1

    return nodes_used


def build_bvh(triangles, tri_normals, tri_uvs, mat_indices):
    """
    python wrapper that initializes memory and runs the jit builder
    """
    assert len(triangles) == len(mat_indices)
    assert len(triangles) == len(tri_normals)
    assert len(triangles) == len(tri_uvs)
    assert len(triangles) > 0

    centroids = np.mean(triangles, axis=1).astype(np.float32)
    tri_ids = np.arange(len(triangles), dtype=np.int32)

    max_nodes = len(triangles) * 2
    nodes = np.zeros((max_nodes, BINS), dtype=np.float32)

    print("Before build_bvh_jit")
    nodes_used = build_bvh_jit(triangles, centroids, tri_ids, nodes)
    print("After build_bvh_jit")

    # trim arrays and reorder geometry according to bvh leaves
    final_nodes = nodes[:nodes_used]
    final_tris = triangles[tri_ids]
    final_norms = tri_normals[tri_ids]
    final_uvs = tri_uvs[tri_ids]
    final_mats = mat_indices[tri_ids]

    return final_nodes, final_tris, final_norms, final_uvs, final_mats
