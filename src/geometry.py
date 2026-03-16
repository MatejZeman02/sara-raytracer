from utils import device_jit
from utils.vec_utils import vec3, normalize, cross, sub, neg, dot
from constants import BARYCENTRIC_EPSILON, ZERO, ONE


@device_jit
def is_valid_normal(n, ray_dir):
    """check if normal is valid (not facing wrong way)"""
    return dot(ray_dir, n) < ZERO


@device_jit
def get_tri_verts(triangles, idx):
    """fetch triangle vertices from array: triangles[idx, abc, xyz]"""
    assert idx >= 0
    a = vec3(triangles[idx, 0, 0], triangles[idx, 0, 1], triangles[idx, 0, 2])
    b = vec3(triangles[idx, 1, 0], triangles[idx, 1, 1], triangles[idx, 1, 2])
    c = vec3(triangles[idx, 2, 0], triangles[idx, 2, 1], triangles[idx, 2, 2])
    return a, b, c


@device_jit
def get_vertex_normals(tri_normals, hit_idx):
    """fetch vertex normals for hit triangle from array: tri_normals[hit_idx, abc, xyz]"""
    na = vec3(
        tri_normals[hit_idx, 0, 0],
        tri_normals[hit_idx, 0, 1],
        tri_normals[hit_idx, 0, 2],
    )
    nb = vec3(
        tri_normals[hit_idx, 1, 0],
        tri_normals[hit_idx, 1, 1],
        tri_normals[hit_idx, 1, 2],
    )
    nc = vec3(
        tri_normals[hit_idx, 2, 0],
        tri_normals[hit_idx, 2, 1],
        tri_normals[hit_idx, 2, 2],
    )
    return na, nb, nc


@device_jit
def compute_surface_normal(triangles, tri_normals, hit_idx, ray_dir, hit_u, hit_v):
    """compute surface normal for hit triangle"""
    # fetch triangle vertices and vertex normals
    a, b, c = get_tri_verts(triangles, hit_idx)
    geom_n = normalize(cross(sub(b, a), sub(c, a)))
    is_backface = not is_valid_normal(geom_n, ray_dir)
    if is_backface:
        geom_n = neg(geom_n)

    na, nb, nc = get_vertex_normals(tri_normals, hit_idx)

    # FIXME: (too late for check?) throw error if vertex normals are missing from obj
    assert (
        na[0] != ZERO or na[1] != ZERO or na[2] != ZERO
    ), "vertex normals are missing from the obj file"
    # 'w' barycentric weight
    w = ONE - hit_u - hit_v
    # verify coordinate sanity
    assert w >= -BARYCENTRIC_EPSILON and w <= ONE + BARYCENTRIC_EPSILON

    # interpolate normal using weights
    interp_x = w * na[0] + hit_u * nb[0] + hit_v * nc[0]
    interp_y = w * na[1] + hit_u * nb[1] + hit_v * nc[1]
    interp_z = w * na[2] + hit_u * nb[2] + hit_v * nc[2]

    n = normalize(vec3(interp_x, interp_y, interp_z))
    if is_backface:
        n = neg(n)

    return a, b, c, na, nb, nc, geom_n, n, w, is_backface
