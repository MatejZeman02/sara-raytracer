import os
import sys

# Ensure src and utils are in the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import numpy as np
from src.bvh import build_bvh
from utils.obj_loader import load_scene
from src.constants import (
    BVH_LEFT_OR_START,
    BVH_RIGHT_OR_COUNT,
    BVH_MIN_X,
    BVH_MIN_Y,
    BVH_MIN_Z,
    BVH_MAX_X,
    BVH_MAX_Y,
    BVH_MAX_Z,
    STACK_SIZE,
)
import time


def intersect_aabb(ray_origin, inv_rd, bmin, bmax):
    """Check if ray intersects AABB. Returns (hit, tmin)."""
    tmin = -np.inf
    tmax = np.inf

    for i in range(3):
        if inv_rd[i] >= 0:
            t1 = (bmin[i] - ray_origin[i]) * inv_rd[i]
            t2 = (bmax[i] - ray_origin[i]) * inv_rd[i]
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
        else:
            t1 = (bmax[i] - ray_origin[i]) * inv_rd[i]
            t2 = (bmin[i] - ray_origin[i]) * inv_rd[i]
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)

    return tmax >= tmin and tmin > 1e-8, tmin


def intersect_triangle(ray_origin, ray_dir, v0, v1, v2):
    """Moller-Trumbore algorithm. Returns (t, u, v) or None."""
    eps = 1e-8
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = np.cross(ray_dir, edge2)
    a = float(np.dot(edge1, h))

    if -eps < a < eps:
        return None

    f = 1.0 / a
    s = ray_origin - v0
    u = float(np.dot(s, h) * f)

    if u < 0.0 or u > 1.0:
        return None

    q = np.cross(s, edge1)
    v = float(np.dot(ray_dir, q) * f)

    if v < 0.0 or u + v > 1.0:
        return None

    t = float(np.dot(edge2, q) * f)

    if t > eps:
        return t, u, v

    return None


def get_closest_hit_cpu(triangles, bvh_nodes, ray_origin, ray_dir):
    """CPU traversal of BVH to find closest hit. Returns dict with stats."""
    closest_t = 1e20
    hit_idx = -1
    tri_tests = 0
    node_tests = 0
    traverse_tests = 0
    max_stack_depth = 0

    stack = np.zeros(STACK_SIZE, dtype=np.int32)
    stack_ptr = 0
    stack[0] = 0

    inv_rd = 1.0 / (ray_dir + 1e-8)

    while stack_ptr >= 0:
        if stack_ptr + 1 > max_stack_depth:
            max_stack_depth = stack_ptr + 1

        node_idx = stack[stack_ptr]
        stack_ptr -= 1
        node_tests += 1

        bmin = np.array(
            [
                bvh_nodes[node_idx, BVH_MIN_X],
                bvh_nodes[node_idx, BVH_MIN_Y],
                bvh_nodes[node_idx, BVH_MIN_Z],
            ],
            dtype=np.float32,
        )

        bmax = np.array(
            [
                bvh_nodes[node_idx, BVH_MAX_X],
                bvh_nodes[node_idx, BVH_MAX_Y],
                bvh_nodes[node_idx, BVH_MAX_Z],
            ],
            dtype=np.float32,
        )

        hit, tmin = intersect_aabb(ray_origin, inv_rd, bmin, bmax)

        if hit and tmin < closest_t:
            traverse_tests += 1
            data1 = bvh_nodes[node_idx, BVH_LEFT_OR_START]
            data2 = bvh_nodes[node_idx, BVH_RIGHT_OR_COUNT]

            if data2 >= 0:
                start = int(data1)
                count = int(data2)
                for i in range(start, start + count):
                    tri_tests += 1
                    v0 = triangles[i, 0, :].astype(np.float32)
                    v1 = triangles[i, 1, :].astype(np.float32)
                    v2 = triangles[i, 2, :].astype(np.float32)

                    result = intersect_triangle(ray_origin, ray_dir, v0, v1, v2)
                    if result is not None:
                        t, u, v = result
                        if 1e-8 < t < closest_t:
                            closest_t = t
                            hit_idx = i
            else:
                left_child = int(data1)
                right_child = int(-data2)
                stack_ptr += 1
                stack[stack_ptr] = right_child
                stack_ptr += 1
                stack[stack_ptr] = left_child

    return {
        "closest_t": closest_t,
        "hit_idx": hit_idx,
        "tri_tests": tri_tests,
        "node_tests": node_tests,
        "traverse_tests": traverse_tests,
        "max_stack_depth": max_stack_depth,
    }


def simulate_traversal(triangles, nodes, cam_data, width=1024, height=1024, num_queries=1000):
    """Simulate a set of ray queries using camera-based rays to measure traversal statistics.
    
    Generates rays from a camera frustum (matching the actual raytracer setup) so that
    the traversal statistics reflect real rendering workload rather than random misses.
    """
    tri_tests_samples = []
    node_tests_samples = []
    query_time_samples = []
    query_depth_samples = []

    # Reconstruct camera setup matching src/setup_vectors.py
    fov = cam_data["fov"]
    origin = np.array(cam_data["pos"], dtype=np.float32)
    dir_vec = np.array(cam_data["dir"], dtype=np.float32)
    up_vec = np.array(cam_data["up"], dtype=np.float32)

    # Compute basis vectors
    b_vec = np.cross(dir_vec, up_vec)
    b_vec = b_vec / np.linalg.norm(b_vec)
    
    t = 1.0  # focal length
    g_w = 2.0 * t * np.tan(fov / 2.0)
    g_h = g_w * (height / width)

    q_w = (g_w / (width - 1)) * b_vec
    q_h = (g_h / (height - 1)) * up_vec
    p00 = t * dir_vec - (g_w / 2.0) * b_vec + (g_h / 2.0) * up_vec

    for i in range(num_queries):
        # Pick a random pixel and generate a ray through the camera frustum
        px = np.random.randint(0, width)
        py = np.random.randint(0, height)

        # Ray direction: p00 + xf*qw - yf*qh (matching src/rays.py compute_primary_ray)
        xf = float(px)
        yf = float(py)
        ray_dir = np.array([
            p00[0] + xf * q_w[0] - yf * q_h[0],
            p00[1] + xf * q_w[1] - yf * q_h[1],
            p00[2] + xf * q_w[2] - yf * q_h[2],
        ], dtype=np.float32)
        ray_dir = ray_dir / np.linalg.norm(ray_dir)
        ray_origin = origin.astype(np.float32)

        start_q = time.perf_counter()
        result = get_closest_hit_cpu(triangles, nodes, ray_origin, ray_dir)
        end_q = time.perf_counter()

        tri_tests_samples.append(result["tri_tests"])
        node_tests_samples.append(result["node_tests"])
        query_time_samples.append((end_q - start_q) * 1000)  # ms
        query_depth_samples.append(result["max_stack_depth"])

    def get_stats(samples):
        return np.min(samples), np.max(samples), np.mean(samples)

    return {
        "tri_tests": get_stats(tri_tests_samples),
        "node_tests": get_stats(node_tests_samples),
        "query_time": get_stats(query_time_samples),
        "query_depth": get_stats(query_depth_samples),
    }


def print_markdown_table(
    construction_time,
    num_internal,
    num_leaves,
    total_nodes,
    num_primitives,
    min_prims,
    max_prims,
    avg_prims,
    max_depth,
    mem_mb,
    stats,
):
    """Print a markdown table for the README."""
    tri_min, tri_max, tri_avg = stats["tri_tests"]
    node_min, node_max, node_avg = stats["node_tests"]
    time_min, time_max, time_avg = stats["query_time"]
    depth_min, depth_max, depth_avg = stats["query_depth"]

    print()
    print("```markdown")
    print("## BVH Performance (Bunny Scene, CPU)")
    print()
    print("| Metric                        | Mean       | Min        | Max         |")
    print("| :---------------------------- | :--------- | :--------- | :---------- |")
    print("| **Construction**              |            |            |             |")
    print(
        f"| Total construction time (s)   | {construction_time:.4f}   |            |             |"
    )
    print(
        f"| Node count                    | {total_nodes}    |            |             |"
    )
    print(
        f"| Internal nodes                | {num_internal}    |            |             |"
    )
    print(
        f"| Leaf nodes                    | {num_leaves}    |            |             |"
    )
    print(
        f"| Number of primitives          | {num_primitives}    |            |             |"
    )
    print(
        f"| Primitives per leaf           | {avg_prims:.2f}     | {min_prims}        | {max_prims}         |"
    )
    print(
        f"| Leaf depth                    | {max_depth}       |            |             |"
    )
    print(
        f"| Memory usage (MB)             | {mem_mb:.2f}     |            |             |"
    )
    print("| **Traversal (per ray)**       |            |            |             |")
    print(
        f"| Incidence operations          | {tri_avg:.2f}     | {tri_min:.2f}     | {tri_max:.2f}      |"
    )
    print(
        f"| Traverse operations           | {node_avg:.2f}     | {node_min:.2f}     | {node_max:.2f}     |"
    )
    print(
        f"| Query time (ms)               | {time_avg:.4f}   | {time_min:.4f}   | {time_max:.4f}  |"
    )
    print(
        f"| Query depth               | {depth_avg:.2f}     | {depth_min:.2f}     | {depth_max:.2f}      |"
    )
    print("```")


def run_metrics():
    # get absolute path to the project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    # Load bunny scene using absolute path
    setup_path = os.path.join(project_root, "scenes", "bunny", "setup.json")

    if not os.path.exists(setup_path):
        print(f"Error: {setup_path} not found.")
        return

    (
        vertices,
        tri_normals,
        tri_uvs,
        mat_indices,
        materials,
        mat_diffuse_tex_ids,
        texture_atlas,
        tex_widths,
        tex_heights,
        light_data,
        cam_data,
    ) = load_scene(setup_path)

    print(f"Loaded {len(vertices) // 9} triangles.")

    # Convert vertices to format expected by build_bvh (shape: N, 3, 3)
    triangles = vertices.astype(np.float32)

    # Construction Metrics
    start_time = time.time()
    nodes, final_tris, final_norms, final_uvs, final_mats = build_bvh(
        triangles, tri_normals, tri_uvs, mat_indices
    )
    end_time = time.time()

    construction_time = end_time - start_time

    # Structural Analysis
    is_leaf = nodes[:, BVH_RIGHT_OR_COUNT] >= 0
    num_leaves = np.sum(is_leaf)
    num_internal = np.sum(~is_leaf)
    total_nodes = len(nodes)
    num_primitives = len(final_tris)

    def get_max_depth(nodes):
        max_depth = 0
        stack = [(0, 1)]
        while stack:
            node_idx, depth = stack.pop()
            if depth > max_depth:
                max_depth = depth
            data1 = nodes[node_idx, BVH_LEFT_OR_START]
            data2 = nodes[node_idx, BVH_RIGHT_OR_COUNT]
            if data2 < 0:
                left_child = int(data1)
                right_child = int(-data2)
                stack.append((left_child, depth + 1))
                stack.append((right_child, depth + 1))
        return max_depth

    max_depth = get_max_depth(nodes)

    # Get primitives per leaf count (BVH_RIGHT_OR_COUNT holds count for leaves)
    leaf_counts = nodes[is_leaf, BVH_RIGHT_OR_COUNT]
    min_prims_per_leaf = int(np.min(leaf_counts)) if num_leaves > 0 else 0
    max_prims_per_leaf = int(np.max(leaf_counts)) if num_leaves > 0 else 0
    avg_prims_per_leaf = float(np.mean(leaf_counts)) if num_leaves > 0 else 0.0

    # Memory usage
    mem_bytes = (
        nodes.nbytes
        + final_tris.nbytes
        + final_norms.nbytes
        + final_uvs.nbytes
        + final_mats.nbytes
    )
    mem_mb = mem_bytes / (1024 * 1024)

    # Traversal Metrics
    # Use the same resolution as the actual raytracer (GPU_DIMENSION from settings)
    resolution = 1024
    stats = simulate_traversal(final_tris, nodes, cam_data, width=resolution, height=resolution, num_queries=1000)

    # Print both human-readable and markdown table formats
    print("\n--- Human-Readable Output ---")
    print(f"Total construction time (s)    : {construction_time:.4f}")
    print(f"Number of internal nodes       : {num_internal}")
    print(f"Number of leaf nodes           : {num_leaves}")
    print(f"Total number of nodes          : {total_nodes}")
    print(f"Number of primitives (tris)    : {num_primitives}")
    print(f"Primitives per leaf (min)      : {min_prims_per_leaf}")
    print(f"Primitives per leaf (max)      : {max_prims_per_leaf}")
    print(f"Primitives per leaf (mean)     : {avg_prims_per_leaf:.2f}")
    print(f"Maximum leaf depth             : {max_depth}")
    print(f"Memory usage (MB)              : {mem_mb:.2f}")

    print("\n--- Traversal Metrics (1000 sample queries) ---")
    tri_min, tri_max, tri_avg = stats["tri_tests"]
    node_min, node_max, node_avg = stats["node_tests"]
    time_min, time_max, time_avg = stats["query_time"]
    depth_min, depth_max, depth_avg = stats["query_depth"]

    print(
        f"Incidence ops per ray (min/max/mean) : {tri_min:.2f}/{tri_max:.2f}/{tri_avg:.2f}"
    )
    print(
        f"Traverse ops per ray (min/max/mean)  : {node_min:.2f}/{node_max:.2f}/{node_avg:.2f}"
    )
    print(
        f"Query time (ms) (min/max/mean)       : {time_min:.4f}/{time_max:.4f}/{time_avg:.4f}"
    )
    print(
        f"Query depth (min/max/mean)       : {depth_min:.2f}/{depth_max:.2f}/{depth_avg:.2f}"
    )

    # Print markdown table for easy copy-paste to README
    print_markdown_table(
        construction_time,
        num_internal,
        num_leaves,
        total_nodes,
        num_primitives,
        min_prims_per_leaf,
        max_prims_per_leaf,
        avg_prims_per_leaf,
        max_depth,
        mem_mb,
        stats,
    )


if __name__ == "__main__":
    run_metrics()
