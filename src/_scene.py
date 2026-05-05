"""Scene loading and BVH construction."""

import json
import os
import numpy as np
from .bvh import build_bvh
from .settings import settings
from utils.obj_loader import load_light_cam_data, load_scene


def _read_color_space(json_file: str) -> str:
    """Return acescg — the renderer operates in a single working space.

    material_color_space is still read from setup.json for cache compatibility,
    but the renderer always converts all materials to acescg at load time.
    """
    return "acescg"


def load_or_build_scene(json_file: str, cache_file: str, t: float) -> tuple:
    """Load scene from cache or build BVH from scratch.

    Returns:
        (triangles, tri_normals, tri_uvs, mat_indices, materials, mat_diffuse_tex_ids,
         diffuse_textures, tex_widths, tex_heights, light_data, cam_data, bvh_nodes, material_color_space)
    """
    required_cache_keys = (
        "bvh_nodes",
        "triangles",
        "tri_normals",
        "tri_uvs",
        "mat_indices",
        "materials",
        "mat_diffuse_tex_ids",
        "diffuse_textures",
        "tex_widths",
        "tex_heights",
    )

    can_use_cache = False
    cache_load_time = 0.0
    if settings.USE_BVH_CACHE and os.path.exists(cache_file):
        import time as _time

        cache_start = _time.perf_counter()
        cache = np.load(cache_file)
        can_use_cache = all(k in cache.files for k in required_cache_keys)
        cache_load_time = _time.perf_counter() - cache_start

    bvh_build_time = 0.0
    if can_use_cache:
        bvh_nodes = cache["bvh_nodes"]
        triangles = cache["triangles"]
        tri_normals = cache["tri_normals"]
        tri_uvs = cache["tri_uvs"]
        mat_indices = cache["mat_indices"]
        materials = cache["materials"]
        mat_diffuse_tex_ids = cache["mat_diffuse_tex_ids"]
        diffuse_textures = cache["diffuse_textures"]
        tex_widths = cache["tex_widths"]
        tex_heights = cache["tex_heights"]
        light_data, cam_data, _, _ = load_light_cam_data(json_file)
    else:
        import time as _time

        (
            triangles,
            tri_normals,
            tri_uvs,
            mat_indices,
            materials,
            mat_diffuse_tex_ids,
            diffuse_textures,
            tex_widths,
            tex_heights,
            light_data,
            cam_data,
        ) = load_scene(json_file)
        assert len(triangles) > 0

        bvh_start = _time.perf_counter()
        bvh_nodes, triangles, tri_normals, tri_uvs, mat_indices = build_bvh(
            triangles,
            tri_normals,
            tri_uvs,
            mat_indices,
            use_sah=settings.USE_SAH,
            use_binning=settings.USE_BINNING,
        )
        bvh_build_time = _time.perf_counter() - bvh_start

        np.savez(
            cache_file,
            bvh_nodes=bvh_nodes,
            triangles=triangles,
            tri_normals=tri_normals,
            tri_uvs=tri_uvs,
            mat_indices=mat_indices,
            materials=materials,
            mat_diffuse_tex_ids=mat_diffuse_tex_ids,
            diffuse_textures=diffuse_textures,
            tex_widths=tex_widths,
            tex_heights=tex_heights,
        )

    material_color_space = _read_color_space(json_file)

    return (
        triangles,
        tri_normals,
        tri_uvs,
        mat_indices,
        materials,
        mat_diffuse_tex_ids,
        diffuse_textures,
        tex_widths,
        tex_heights,
        light_data,
        cam_data,
        bvh_nodes,
        material_color_space,
        bvh_build_time,
    )
