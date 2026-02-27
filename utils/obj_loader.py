import os
import json

import numpy as np  # type: ignore

# custom compiled Python wrapper for c++ tinyobjloader
from . import tinyobjloader_py as tiny_obj_loader


def load_camera_txt(txt_path):
    """Deprecated: Loads camera configuration from the .txt file."""
    with open(txt_path, "r") as f:
        data = json.load(f)

    cam_data = {
        "obj_file": data.get("obj_file", ""),
        "pos": np.array(data.get("pos", [0, 0, 0]), dtype=np.float32),
        "up": np.array(data.get("up", [0, 1, 0]), dtype=np.float32),
        "dir": np.array(data.get("dir", [0, 0, -1]), dtype=np.float32),
        "fov": float(data.get("fov", 0.78)),
    }

    return cam_data


def load_mtl(filename):
    """Loads materials including emission (Ke)."""
    materials = {}
    current_mat = None

    if not os.path.exists(filename):
        return materials

    with open(filename, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0] == "#":
                continue

            if parts[0] == "newmtl":
                current_mat = parts[1]
                materials[current_mat] = {
                    "Kd": [0.8, 0.8, 0.8],
                    "Ks": [0.0, 0.0, 0.0],
                    "Ns": 10.0,
                    "Ke": [0.0, 0.0, 0.0],  # Emission
                }
            elif parts[0] == "Kd" and current_mat:
                materials[current_mat]["Kd"] = [float(x) for x in parts[1:4]]
            elif parts[0] == "Ks" and current_mat:
                materials[current_mat]["Ks"] = [float(x) for x in parts[1:4]]
            elif parts[0] == "Ke" and current_mat:
                materials[current_mat]["Ke"] = [float(x) for x in parts[1:4]]
            elif parts[0] == "Ns" and current_mat:
                materials[current_mat]["Ns"] = float(parts[1])

    return materials


def load_scene_old(txt_path):
    """
    Deprecated!!!
    loads the entire scene based on the .txt file.
    """
    base_dir = os.path.dirname(txt_path)
    cam_data = load_camera_txt(txt_path)

    obj_path = os.path.join(base_dir, cam_data["obj_file"])

    vertices = []
    triangles = []
    material_indices = []

    materials_dict = {}
    mat_name_to_idx = {}
    current_mat_idx = -1

    with open(obj_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0] == "#":
                continue

            if parts[0] == "mtllib":
                mtl_path = os.path.join(base_dir, parts[1])
                materials_dict.update(load_mtl(mtl_path))
                for i, mat_name in enumerate(materials_dict.keys()):
                    mat_name_to_idx[mat_name] = i

            elif parts[0] == "usemtl":
                current_mat_idx = mat_name_to_idx.get(parts[1], -1)

            elif parts[0] == "v":
                vertices.append([float(x) for x in parts[1:4]])

            elif parts[0] == "f":
                # Handle relative negative indices and N-gons (triangulation)
                face_verts = []
                for p in parts[1:]:
                    idx_str = p.split("/")[0]
                    idx = int(idx_str)
                    # Convert negative relative index to absolute 0-based index
                    if idx < 0:
                        idx = len(vertices) + idx
                    else:
                        idx = idx - 1
                    face_verts.append(vertices[idx])

                # Simple fan triangulation for quads/polygons
                for i in range(1, len(face_verts) - 1):
                    triangles.append([face_verts[0], face_verts[i], face_verts[i + 1]])
                    material_indices.append(current_mat_idx)

    triangles_np = np.array(triangles, dtype=np.float32)
    mat_indices_np = np.array(material_indices, dtype=np.int32)

    # Pack materials: [Kd(3), Ks(3), Ns(1), Ke(3)] -> 10 floats per material
    mat_array = np.zeros((max(1, len(mat_name_to_idx)), 10), dtype=np.float32)
    for mat_name, idx in mat_name_to_idx.items():
        mat = materials_dict[mat_name]
        mat_array[idx, 0:3] = mat["Kd"]
        mat_array[idx, 3:6] = mat["Ks"]
        mat_array[idx, 6] = mat["Ns"]
        mat_array[idx, 7:10] = mat["Ke"]

    return triangles_np, mat_indices_np, mat_array, cam_data


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
