from utils import device_jit
from numpy import float32
from math import sqrt
from utils.vec_utils import vec3, add, sub, mul, neg, dot
from shading import cook_torrance_shading
from traversal import is_in_shadow
from rays import compute_inv_dir
from constants import (
    ZERO,
    ONE,
    TWO,
    RAY_EPSILON,
    MAT_DIFFUSE_R,
    MAT_DIFFUSE_G,
    MAT_DIFFUSE_B,
    MAT_SPECULAR_R,
    MAT_SPECULAR_G,
    MAT_SPECULAR_B,
    MAT_ROUGHNESS,
    MAT_EMISSIVE_R,
    MAT_EMISSIVE_G,
    MAT_EMISSIVE_B,
)


@device_jit
def get_emissive_color(materials, mat_idx):
    return vec3(
        materials[mat_idx, MAT_EMISSIVE_R],
        materials[mat_idx, MAT_EMISSIVE_G],
        materials[mat_idx, MAT_EMISSIVE_B],
    )


@device_jit
def compute_chiang_offset_vec(a, b, c, na, nb, nc, w, hit_u, hit_v, p, is_backface):
    """CHIANG'S analytical terminator offset @Disney 2019"""
    # return zero vector if normals missing
    if na[0] == ZERO and na[1] == ZERO and na[2] == ZERO:
        return vec3(ZERO, ZERO, ZERO)

    fa_ = neg(na) if is_backface else na  # face_normal_a
    fb_ = neg(nb) if is_backface else nb  # face_normal_b
    fc_ = neg(nc) if is_backface else nc  # face_normal_c

    p_a = add(p, mul(fa_, dot(sub(a, p), fa_)))
    p_b = add(p, mul(fb_, dot(sub(b, p), fb_)))
    p_c = add(p, mul(fc_, dot(sub(c, p), fc_)))

    p_curve = add(add(mul(p_a, w), mul(p_b, hit_u)), mul(p_c, hit_v))
    return sub(p_curve, p)


@device_jit
def compute_shadow_ray_origin(
    a, b, c, na, nb, nc, w, hit_u, hit_v, p, geom_n, is_backface
):
    # apply standard epsilon offset along the correct face direction
    offset_n = neg(geom_n) if is_backface else geom_n
    base_p = add(p, mul(offset_n, RAY_EPSILON))

    chiang_vec = compute_chiang_offset_vec(
        a, b, c, na, nb, nc, w, hit_u, hit_v, p, is_backface
    )

    sq_len = dot(chiang_vec, chiang_vec)
    max_offset = float32(0.01)  # 1cm
    max_sq = max_offset * max_offset

    if sq_len > max_sq:
        # ensure length is valid before root calculation
        assert sq_len > ZERO
        length = float32(sqrt(sq_len))
        normalized_chiang = mul(chiang_vec, ONE / length)
        chiang_vec = mul(normalized_chiang, max_offset)

    return add(base_p, chiang_vec)


@device_jit
def compute_shadowed(
    triangles,
    bvh_nodes,
    use_bvh,
    a,
    b,
    c,
    na,
    nb,
    nc,
    w,
    hit_u,
    hit_v,
    p,
    geom_n,
    is_backface,
    d_l,
    dist_to_light,
    stack,
):
    """compute if hit point is shadowed by tracing a ray towards the light source"""
    shadow_ro = compute_shadow_ray_origin(
        a, b, c, na, nb, nc, w, hit_u, hit_v, p, geom_n, is_backface
    )

    inv_dl = compute_inv_dir(d_l)
    # shorten the ray target distance to avoid hitting the emissive triangle itself
    safe_dist = dist_to_light - RAY_EPSILON
    assert (
        safe_dist > ZERO
    ), "Light is too close to the surface, increase RAY_EPSILON or move the light further away"
    return is_in_shadow(
        triangles, bvh_nodes, use_bvh, shadow_ro, d_l, inv_dl, safe_dist, stack
    )


@device_jit
def compute_lit_color(materials, mat_idx, light_color, n, v_vec, d_l, shadowed):
    """Shading"""
    # fetch material properties
    r_d = vec3(  # diffuse_reflectance
        materials[mat_idx, MAT_DIFFUSE_R],
        materials[mat_idx, MAT_DIFFUSE_G],
        materials[mat_idx, MAT_DIFFUSE_B],
    )
    r_s = vec3(  # specular_reflectance
        materials[mat_idx, MAT_SPECULAR_R],
        materials[mat_idx, MAT_SPECULAR_G],
        materials[mat_idx, MAT_SPECULAR_B],
    )
    h_val = materials[mat_idx, MAT_ROUGHNESS]  # roughness_or_shininess
    l_color = vec3(light_color[0], light_color[1], light_color[2])  # light_radiance

    if shadowed:
        return vec3(ZERO, ZERO, ZERO)
    return cook_torrance_shading(n, v_vec, d_l, r_d, r_s, h_val, l_color)
