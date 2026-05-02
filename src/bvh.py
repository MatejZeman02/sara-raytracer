"""sa (surface area heuristic) bvh builder for ray tracing on cpu with jit using stack"""

import numpy as np
from numba import njit
from .constants import (
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
MAX_VAL = np.float32(1e20)


@njit
def get_aabb(triangles, tri_ids, start, end):
    """bounding box for a subset of triangles"""
    assert start < end

    bmin = np.array([MAX_VAL, MAX_VAL, MAX_VAL], dtype=np.float32)
    bmax = np.array([-MAX_VAL, -MAX_VAL, -MAX_VAL], dtype=np.float32)

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
def expand_triangle_bounds(triangles, t_idx, out_min, out_max):
    """expand aabb bounds with all vertices of one triangle."""
    for j in range(3):
        v_x = triangles[t_idx, j, 0]
        v_y = triangles[t_idx, j, 1]
        v_z = triangles[t_idx, j, 2]

        if v_x < out_min[0]:
            out_min[0] = v_x
        if v_x > out_max[0]:
            out_max[0] = v_x

        if v_y < out_min[1]:
            out_min[1] = v_y
        if v_y > out_max[1]:
            out_max[1] = v_y

        if v_z < out_min[2]:
            out_min[2] = v_z
        if v_z > out_max[2]:
            out_max[2] = v_z


@njit
def sah_split_search(triangles, centroids, tri_ids, start, end, bmin, bmax, cmin, cmax, area_p):
    """find best split using sah cost evaluation over all possible split positions.
    iterates all triangles sorted by centroid on each axis.
    returns (best_axis, best_split, best_cost).
    """
    best_cost = MAX_VAL
    best_axis = -1
    best_split = 0.0

    n = end - start
    if n <= 2:
        return best_axis, best_split, best_cost

    # temporary arrays for sorting
    sort_keys = np.zeros(n, dtype=np.float32)
    sort_ids = np.zeros(n, dtype=np.int32)

    for axis in range(3):
        if cmin[axis] == cmax[axis]:
            continue

        extent = cmax[axis] - cmin[axis]
        if extent <= 0.0:
            continue

        # copy and sort by centroid on this axis
        for i in range(n):
            idx = start + i
            sort_keys[i] = centroids[tri_ids[idx], axis]
            sort_ids[i] = tri_ids[idx]

        # insertion sort on sort_keys
        for i in range(1, n):
            key = sort_keys[i]
            val = sort_ids[i]
            j = i - 1
            while j >= 0 and sort_keys[j] > key:
                sort_keys[j + 1] = sort_keys[j]
                sort_ids[j + 1] = sort_ids[j]
                j -= 1
            sort_keys[j + 1] = key
            sort_ids[j + 1] = val

        # sweep and evaluate sah cost at each unique split position
        for split in range(1, n):
            # compute left bbox
            l_bmin = np.array([MAX_VAL, MAX_VAL, MAX_VAL], dtype=np.float32)
            l_bmax = np.array([-MAX_VAL, -MAX_VAL, -MAX_VAL], dtype=np.float32)
            for i in range(split):
                expand_triangle_bounds(triangles, sort_ids[i], l_bmin, l_bmax)

            # compute right bbox
            r_bmin = np.array([MAX_VAL, MAX_VAL, MAX_VAL], dtype=np.float32)
            r_bmax = np.array([-MAX_VAL, -MAX_VAL, -MAX_VAL], dtype=np.float32)
            for i in range(split, n):
                expand_triangle_bounds(triangles, sort_ids[i], r_bmin, r_bmax)

            nl = split
            nr = n - split
            if nl == 0 or nr == 0:
                continue

            la = get_area(l_bmin, l_bmax)
            ra = get_area(r_bmin, r_bmax)
            cost = 1.0 + (nl * la + nr * ra) / area_p

            if cost < best_cost:
                best_cost = cost
                best_axis = axis
                # Return a centroid threshold so the later in-place partition
                # matches the split that was actually evaluated here.
                best_split = np.float32(0.5 * (sort_keys[split - 1] + sort_keys[split]))

    return best_axis, best_split, best_cost


@njit
def median_split(triangles, centroids, tri_ids, start, end, bmin, bmax):
    """find best split using median split on largest bbox extent axis."""
    # find axis with largest bbox extent
    best_axis = 0
    best_extent = bmax[0] - bmin[0]
    for axis in range(1, 3):
        extent = bmax[axis] - bmin[axis]
        if extent > best_extent:
            best_extent = extent
            best_axis = axis

    if best_extent <= 0.0:
        return -1, 0.0

    # sort triangles by centroid on this axis using insertion sort
    n = end - start
    sort_keys = np.zeros(n, dtype=np.float32)
    sort_ids = np.zeros(n, dtype=np.int32)
    for i in range(n):
        idx = start + i
        sort_keys[i] = centroids[tri_ids[idx], best_axis]
        sort_ids[i] = tri_ids[idx]

    for i in range(1, n):
        key = sort_keys[i]
        val = sort_ids[i]
        j = i - 1
        while j >= 0 and sort_keys[j] > key:
            sort_keys[j + 1] = sort_keys[j]
            sort_ids[j + 1] = sort_ids[j]
            j -= 1
        sort_keys[j + 1] = key
        sort_ids[j + 1] = val

    # median split: split in the middle
    split_idx = n // 2
    best_split = np.float32(0.5 * (sort_keys[split_idx - 1] + sort_keys[split_idx]))

    return best_axis, best_split


@njit
def build_bvh_jit(triangles, centroids, tri_ids, nodes, use_sah, use_binning):
    """
    iterative bvh builder using stack.
    supports sah and binning toggles for performance comparison.

    parameters:
        use_sah: if true, use surface area heuristic for split selection
        use_binning: if true, use centroid binning for split candidate generation
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
        cmin = np.array([MAX_VAL, MAX_VAL, MAX_VAL], dtype=np.float32)
        cmax = np.array([-MAX_VAL, -MAX_VAL, -MAX_VAL], dtype=np.float32)
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

        area_p = get_area(bmin, bmax)
        if area_p <= 0.0:
            nodes[node_idx, BVH_MIN_X : BVH_MIN_Z + 1] = bmin
            nodes[node_idx, BVH_MAX_X : BVH_MAX_Z + 1] = bmax
            nodes[node_idx, BVH_LEFT_OR_START] = start
            nodes[node_idx, BVH_RIGHT_OR_COUNT] = num_tris
            continue

        best_axis = -1
        best_split = 0.0
        best_cost = MAX_VAL

        if use_sah:
            # sah-based split selection
            if use_binning:
                # true binned sah: one histogram build per axis, o(bins) split sweep
                for axis in range(3):
                    if cmin[axis] == cmax[axis]:
                        continue

                    scale = BINS / (cmax[axis] - cmin[axis])

                    # bin histogram (count + bin aabbs)
                    bin_count = np.zeros(BINS, dtype=np.int32)
                    bin_min = np.full((BINS, 3), MAX_VAL, dtype=np.float32)
                    bin_max = np.full((BINS, 3), -MAX_VAL, dtype=np.float32)

                    for i in range(start, end):
                        t_idx = tri_ids[i]
                        bin_idx = int((centroids[t_idx, axis] - cmin[axis]) * scale)
                        if bin_idx < 0:
                            bin_idx = 0
                        elif bin_idx >= BINS:
                            bin_idx = BINS - 1

                        bin_count[bin_idx] += 1
                        expand_triangle_bounds(
                            triangles, t_idx, bin_min[bin_idx], bin_max[bin_idx]
                        )

                    # prefix (left) accumulations
                    left_count = np.zeros(BINS - 1, dtype=np.int32)
                    left_area = np.zeros(BINS - 1, dtype=np.float32)
                    run_count = 0
                    run_min = np.array([MAX_VAL, MAX_VAL, MAX_VAL], dtype=np.float32)
                    run_max = np.array([-MAX_VAL, -MAX_VAL, -MAX_VAL], dtype=np.float32)

                    for b in range(BINS - 1):
                        if bin_count[b] > 0:
                            run_count += bin_count[b]

                            if bin_min[b, 0] < run_min[0]:
                                run_min[0] = bin_min[b, 0]
                            if bin_max[b, 0] > run_max[0]:
                                run_max[0] = bin_max[b, 0]

                            if bin_min[b, 1] < run_min[1]:
                                run_min[1] = bin_min[b, 1]
                            if bin_max[b, 1] > run_max[1]:
                                run_max[1] = bin_max[b, 1]

                            if bin_min[b, 2] < run_min[2]:
                                run_min[2] = bin_min[b, 2]
                            if bin_max[b, 2] > run_max[2]:
                                run_max[2] = bin_max[b, 2]

                        left_count[b] = run_count
                        left_area[b] = get_area(run_min, run_max)

                    # suffix (right) accumulations
                    right_count = np.zeros(BINS - 1, dtype=np.int32)
                    right_area = np.zeros(BINS - 1, dtype=np.float32)
                    run_count = 0
                    run_min = np.array([MAX_VAL, MAX_VAL, MAX_VAL], dtype=np.float32)
                    run_max = np.array([-MAX_VAL, -MAX_VAL, -MAX_VAL], dtype=np.float32)

                    for b in range(BINS - 1, 0, -1):
                        if bin_count[b] > 0:
                            run_count += bin_count[b]

                            if bin_min[b, 0] < run_min[0]:
                                run_min[0] = bin_min[b, 0]
                            if bin_max[b, 0] > run_max[0]:
                                run_max[0] = bin_max[b, 0]

                            if bin_min[b, 1] < run_min[1]:
                                run_min[1] = bin_min[b, 1]
                            if bin_max[b, 1] > run_max[1]:
                                run_max[1] = bin_max[b, 1]

                            if bin_min[b, 2] < run_min[2]:
                                run_min[2] = bin_min[b, 2]
                            if bin_max[b, 2] > run_max[2]:
                                run_max[2] = bin_max[b, 2]

                        right_count[b - 1] = run_count
                        right_area[b - 1] = get_area(run_min, run_max)

                    # evaluate splits in o(bins)
                    for b in range(BINS - 1):
                        nl = left_count[b]
                        nr = right_count[b]
                        if nl == 0 or nr == 0:
                            continue

                        cost = 1.0 + (nl * left_area[b] + nr * right_area[b]) / area_p
                        if cost < best_cost:
                            best_cost = cost
                            best_axis = axis
                            best_split = cmin[axis] + (b + 1) / scale
            else:
                # sah without binning: iterate all triangles to find best split
                result_axis, result_split, result_cost = sah_split_search(
                    triangles, centroids, tri_ids, start, end, bmin, bmax, cmin, cmax, area_p
                )
                if result_axis >= 0 and result_cost < best_cost:
                    best_axis = result_axis
                    best_split = result_split
                    best_cost = result_cost
        else:
            # no sah: use median split on largest bbox extent axis
            result_axis, result_split = median_split(
                triangles, centroids, tri_ids, start, end, bmin, bmax
            )
            if result_axis >= 0:
                best_axis = result_axis
                best_split = result_split

        # make leaf if sah cost is worse than just intersecting all (only for sah mode)
        if use_sah and best_cost >= num_tris:
            nodes[node_idx, BVH_MIN_X : BVH_MIN_Z + 1] = bmin
            nodes[node_idx, BVH_MAX_X : BVH_MAX_Z + 1] = bmax
            nodes[node_idx, BVH_LEFT_OR_START] = start
            nodes[node_idx, BVH_RIGHT_OR_COUNT] = num_tris
            continue

        # make leaf if no valid split found
        if best_axis < 0:
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


def build_bvh(triangles, tri_normals, tri_uvs, mat_indices, use_sah=True, use_binning=True):
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
    nodes = np.zeros((max_nodes, BVH_RIGHT_OR_COUNT + 1), dtype=np.float32)

    nodes_used = build_bvh_jit(triangles, centroids, tri_ids, nodes, use_sah, use_binning)

    # trim arrays and reorder geometry according to bvh leaves
    final_nodes = nodes[:nodes_used]
    final_tris = triangles[tri_ids]
    final_norms = tri_normals[tri_ids]
    final_uvs = tri_uvs[tri_ids]
    final_mats = mat_indices[tri_ids]

    return final_nodes, final_tris, final_norms, final_uvs, final_mats
