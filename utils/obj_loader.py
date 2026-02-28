import os
import json

import numpy as np

# custom compiled Python wrapper for c++ tinyobjloader
from . import tinyobjloader_py as tiny_obj_loader


def load_scene(txt_path):
    """
    loads the scene using index-based c++ binding and converts it for the kernel
    """
    base_dir = os.path.dirname(txt_path)
    with open(txt_path, "r") as f:
        cam_data = json.load(f)
    obj_path = os.path.join(base_dir, cam_data["obj_file"])

    assert os.path.exists(obj_path)

    # call c++ binding
    # returns: (flat_vertices, flat_indices, mat_ids, flat_materials)
    raw_verts, raw_indices, raw_mat_idx, raw_mats = tiny_obj_loader.load_obj(
        obj_path, base_dir
    )

    # convert to numpy arrays
    # vertices: (n_verts, 3)
    vertices = np.array(raw_verts, dtype=np.float32).reshape(-1, 3)
    # indices: (n_tris * 3)
    indices = np.array(raw_indices, dtype=np.int32)

    # ensure we have valid geometry
    assert len(indices) % 3 == 0
    assert len(vertices) > 0

    # reconstruction of triangles using fancy indexing
    # this creates the (n, 3, 3) array needed for the current cuda kernel
    triangles = vertices[indices].reshape(-1, 3, 3)

    mat_indices = np.array(raw_mat_idx, dtype=np.int32)
    materials = np.array(raw_mats, dtype=np.float32).reshape(-1, 10)

    return triangles, mat_indices, materials, cam_data
