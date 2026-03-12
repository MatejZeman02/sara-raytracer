"""Uses tinyobjloader to load the obj file and extract the geometry and materials."""

import os
import json

import numpy as np
from numba import njit

from . import tinyobjloader_py as tiny_obj_loader


@njit
def _build_geometry(vertices, normals, v_idx, n_idx, num_tris):
    # build 3d arrays for triangles and normals from flat buffers
    assert len(v_idx) == num_tris * 3
    assert len(n_idx) == num_tris * 3

    out_tris = np.zeros((num_tris, 3, 3), dtype=np.float32)
    out_norms = np.zeros((num_tris, 3, 3), dtype=np.float32)

    for i in range(num_tris):
        for j in range(3):
            # extract vertex positions
            vi = v_idx[i * 3 + j]
            out_tris[i, j, 0] = vertices[vi * 3 + 0]
            out_tris[i, j, 1] = vertices[vi * 3 + 1]
            out_tris[i, j, 2] = vertices[vi * 3 + 2]

            # extract vertex normals if they exist
            ni = n_idx[i * 3 + j]
            if ni >= 0 and len(normals) > 0:
                out_norms[i, j, 0] = normals[ni * 3 + 0]
                out_norms[i, j, 1] = normals[ni * 3 + 1]
                out_norms[i, j, 2] = normals[ni * 3 + 2]

    return out_tris, out_norms


def load_light_cam_data(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data["light"], data["camera"], data["obj_file"]


def load_scene(txt_path):
    # loads the scene using custom c++ binding
    base_dir = os.path.dirname(txt_path)
    light_data, cam_data, obj_file = load_light_cam_data(txt_path)
    obj_path = os.path.join(base_dir, obj_file)

    assert os.path.exists(obj_path)

    # call c++ binding
    raw_verts, raw_norms, raw_v_idx, raw_n_idx, raw_mat_idx, raw_mats = (
        tiny_obj_loader.load_obj(obj_path, base_dir)
    )

    vertices = np.array(raw_verts, dtype=np.float32)
    normals = np.array(raw_norms, dtype=np.float32)
    v_idx = np.array(raw_v_idx, dtype=np.int32)
    n_idx = np.array(raw_n_idx, dtype=np.int32)

    assert len(v_idx) % 3 == 0

    mat_indices = np.array(raw_mat_idx, dtype=np.int32)
    materials = np.array(raw_mats, dtype=np.float32).reshape(-1, 14)

    num_tris = len(mat_indices)
    assert num_tris > 0

    # build final geometry using numba
    triangles, tri_normals = _build_geometry(vertices, normals, v_idx, n_idx, num_tris)

    return triangles, tri_normals, mat_indices, materials, light_data, cam_data
