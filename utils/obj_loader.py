"""Uses tinyobjloader to load the obj file and extract the geometry and materials."""

import os
import json

import numpy as np
from numba import njit
from PIL import Image

from src.constants import (
    NO_TEXTURE,
    MAT_DIFFUSE_R,
    MAT_DIFFUSE_G,
    MAT_DIFFUSE_B,
    MAT_SPECULAR_R,
    MAT_SPECULAR_G,
    MAT_SPECULAR_B,
    MAT_EMISSIVE_R,
    MAT_EMISSIVE_G,
    MAT_EMISSIVE_B,
    MAT_TRANSMISSION_R,
    MAT_TRANSMISSION_G,
    MAT_TRANSMISSION_B,
)

from . import tinyobjloader_py as tiny_obj_loader


@njit
def _build_geometry(vertices, normals, texcoords, v_idx, n_idx, t_idx, num_tris):
    # build 3d arrays for triangles, normals and UVs from flat buffers
    assert len(v_idx) == num_tris * 3
    assert len(n_idx) == num_tris * 3
    assert len(t_idx) == num_tris * 3

    out_tris = np.zeros((num_tris, 3, 3), dtype=np.float32)
    out_norms = np.zeros((num_tris, 3, 3), dtype=np.float32)
    out_uvs = np.zeros((num_tris, 3, 2), dtype=np.float32)

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

            # extract texture coordinates if they exist
            ti = t_idx[i * 3 + j]
            if ti >= 0 and len(texcoords) > 0:
                out_uvs[i, j, 0] = texcoords[ti * 2 + 0]
                out_uvs[i, j, 1] = texcoords[ti * 2 + 1]

    return out_tris, out_norms, out_uvs


def _build_texture_data(diffuse_texnames, base_dir):
    # map material -> texture id, and build a padded texture atlas for device kernels
    num_materials = len(diffuse_texnames)
    mat_diffuse_tex_ids = np.full(num_materials, NO_TEXTURE, dtype=np.int32)

    path_to_tex_id = {}
    texture_images = []
    texture_widths = []
    texture_heights = []

    for mat_idx, tex_name in enumerate(diffuse_texnames):
        if tex_name is None or len(tex_name.strip()) == 0:
            continue

        tex_path = os.path.join(base_dir, tex_name)
        if not os.path.exists(tex_path):
            continue

        if tex_path not in path_to_tex_id:
            tex_id = len(texture_images)
            path_to_tex_id[tex_path] = tex_id

            tex_img = Image.open(tex_path).convert("RGB")
            tex_arr = np.asarray(tex_img, dtype=np.float32) / 255.0
            texture_images.append(tex_arr)
            texture_heights.append(tex_arr.shape[0])
            texture_widths.append(tex_arr.shape[1])

        mat_diffuse_tex_ids[mat_idx] = path_to_tex_id[tex_path]

    if len(texture_images) == 0:
        texture_atlas = np.zeros((1, 1, 1, 3), dtype=np.float32)
        tex_widths = np.ones(1, dtype=np.int32)
        tex_heights = np.ones(1, dtype=np.int32)
        return mat_diffuse_tex_ids, texture_atlas, tex_widths, tex_heights

    num_textures = len(texture_images)
    max_h = max(texture_heights)
    max_w = max(texture_widths)

    texture_atlas = np.zeros((num_textures, max_h, max_w, 3), dtype=np.float32)
    tex_widths = np.array(texture_widths, dtype=np.int32)
    tex_heights = np.array(texture_heights, dtype=np.int32)

    for tex_id in range(num_textures):
        h = tex_heights[tex_id]
        w = tex_widths[tex_id]
        texture_atlas[tex_id, :h, :w, :] = texture_images[tex_id]

    return mat_diffuse_tex_ids, texture_atlas, tex_widths, tex_heights


def _srgb_to_linear(rgb):
    rgb = np.maximum(rgb, 0.0)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def _linear_srgb_to_acescg(rgb):
    # Matrix from linear sRGB (Rec.709 D65) to ACEScg (AP1 D60).
    mat = np.array(
        [
            [0.613097, 0.339523, 0.047379],
            [0.070194, 0.916355, 0.013451],
            [0.020615, 0.109569, 0.869816],
        ],
        dtype=np.float32,
    )
    return rgb @ mat.T


def _convert_material_colors_to_acescg(materials):
    if materials.size == 0:
        return materials

    def convert_slice(start_idx):
        rgb = materials[:, start_idx : start_idx + 3]
        rgb_lin = _srgb_to_linear(rgb)
        materials[:, start_idx : start_idx + 3] = _linear_srgb_to_acescg(rgb_lin)

    convert_slice(MAT_DIFFUSE_R)
    convert_slice(MAT_SPECULAR_R)
    convert_slice(MAT_EMISSIVE_R)
    convert_slice(MAT_TRANSMISSION_R)
    return materials


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
    (
        raw_verts,
        raw_norms,
        raw_texcoords,
        raw_v_idx,
        raw_n_idx,
        raw_t_idx,
        raw_mat_idx,
        raw_mats,
        raw_diffuse_texnames,
    ) = tiny_obj_loader.load_obj(obj_path, base_dir)

    vertices = np.array(raw_verts, dtype=np.float32)
    normals = np.array(raw_norms, dtype=np.float32)
    texcoords = np.array(raw_texcoords, dtype=np.float32)
    v_idx = np.array(raw_v_idx, dtype=np.int32)
    n_idx = np.array(raw_n_idx, dtype=np.int32)
    t_idx = np.array(raw_t_idx, dtype=np.int32)

    assert len(v_idx) % 3 == 0

    mat_indices = np.array(raw_mat_idx, dtype=np.int32)
    materials = np.array(raw_mats, dtype=np.float32).reshape(-1, 14)
    materials = _convert_material_colors_to_acescg(materials)
    mat_diffuse_tex_ids, texture_atlas, tex_widths, tex_heights = _build_texture_data(
        raw_diffuse_texnames, base_dir
    )

    num_tris = len(mat_indices)
    assert num_tris > 0

    # build final geometry using numba
    triangles, tri_normals, tri_uvs = _build_geometry(
        vertices, normals, texcoords, v_idx, n_idx, t_idx, num_tris
    )

    return (
        triangles,
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
    )
