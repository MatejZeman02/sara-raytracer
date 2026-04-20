from utils import device_jit
from numpy import float32
from math import sqrt, floor
from utils.vec_utils import vec3, add, sub, mul, neg, dot
from shading import cook_torrance_shading
from traversal import is_in_shadow
from rays import compute_inv_dir
from constants import (
    ZERO,
    ONE,
    TWO,
    DIST_TO_LIGHT_MULT,
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
    NO_TEXTURE,
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

    # Chiang offset must never move the shadow origin back into the surface.
    normal_comp = dot(chiang_vec, offset_n)
    if normal_comp < ZERO:
        chiang_vec = sub(chiang_vec, mul(offset_n, normal_comp))

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
    # shorter ray target distance avoids hitting the emissive triangle itself
    tmax_guard = max(DIST_TO_LIGHT_MULT * dist_to_light, RAY_EPSILON)
    safe_dist = dist_to_light - tmax_guard
    assert (
        safe_dist > ZERO
    ), "Light is too close to the surface, increase RAY_EPSILON or move the light further away"
    return is_in_shadow(
        triangles, bvh_nodes, use_bvh, shadow_ro, d_l, inv_dl, safe_dist, stack
    )


@device_jit
def sample_diffuse_texture(
    mat_idx,
    hit_idx,
    hit_u,
    hit_v,
    tri_uvs,
    mat_diffuse_tex_ids,
    diffuse_textures,
    tex_widths,
    tex_heights,
):
    tex_id = mat_diffuse_tex_ids[mat_idx]
    if tex_id == NO_TEXTURE:
        return vec3(ONE, ONE, ONE)

    w = ONE - hit_u - hit_v
    uv_u = (
        w * tri_uvs[hit_idx, 0, 0]
        + hit_u * tri_uvs[hit_idx, 1, 0]
        + hit_v * tri_uvs[hit_idx, 2, 0]
    )
    uv_v = (
        w * tri_uvs[hit_idx, 0, 1]
        + hit_u * tri_uvs[hit_idx, 1, 1]
        + hit_v * tri_uvs[hit_idx, 2, 1]
    )

    # repeat mode in both UV axes
    uv_u = uv_u - float32(floor(uv_u))
    uv_v = uv_v - float32(floor(uv_v))

    tex_w = tex_widths[tex_id]
    tex_h = tex_heights[tex_id]
    if tex_w <= 0 or tex_h <= 0:
        return vec3(ONE, ONE, ONE)

    x = uv_u * float32(tex_w - 1)
    # image data is top-left based, UV v is bottom-left based
    y = (ONE - uv_v) * float32(tex_h - 1)

    x0 = int(floor(x))
    y0 = int(floor(y))
    x1 = (x0 + 1) % tex_w
    y1 = (y0 + 1) % tex_h

    tx = x - float32(x0)
    ty = y - float32(y0)

    c00_r = diffuse_textures[tex_id, y0, x0, 0]
    c00_g = diffuse_textures[tex_id, y0, x0, 1]
    c00_b = diffuse_textures[tex_id, y0, x0, 2]

    c10_r = diffuse_textures[tex_id, y0, x1, 0]
    c10_g = diffuse_textures[tex_id, y0, x1, 1]
    c10_b = diffuse_textures[tex_id, y0, x1, 2]

    c01_r = diffuse_textures[tex_id, y1, x0, 0]
    c01_g = diffuse_textures[tex_id, y1, x0, 1]
    c01_b = diffuse_textures[tex_id, y1, x0, 2]

    c11_r = diffuse_textures[tex_id, y1, x1, 0]
    c11_g = diffuse_textures[tex_id, y1, x1, 1]
    c11_b = diffuse_textures[tex_id, y1, x1, 2]

    top_r = c00_r * (ONE - tx) + c10_r * tx
    top_g = c00_g * (ONE - tx) + c10_g * tx
    top_b = c00_b * (ONE - tx) + c10_b * tx

    bot_r = c01_r * (ONE - tx) + c11_r * tx
    bot_g = c01_g * (ONE - tx) + c11_g * tx
    bot_b = c01_b * (ONE - tx) + c11_b * tx

    return vec3(
        top_r * (ONE - ty) + bot_r * ty,
        top_g * (ONE - ty) + bot_g * ty,
        top_b * (ONE - ty) + bot_b * ty,
    )


@device_jit
def compute_lit_color(
    materials,
    mat_idx,
    hit_idx,
    hit_u,
    hit_v,
    tri_uvs,
    mat_diffuse_tex_ids,
    diffuse_textures,
    tex_widths,
    tex_heights,
    light_color,
    n,
    v_vec,
    d_l,
    shadowed,
):
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

    # texture modulates only diffuse reflectance
    tex_color = sample_diffuse_texture(
        mat_idx,
        hit_idx,
        hit_u,
        hit_v,
        tri_uvs,
        mat_diffuse_tex_ids,
        diffuse_textures,
        tex_widths,
        tex_heights,
    )
    r_d = vec3(r_d[0] * tex_color[0], r_d[1] * tex_color[1], r_d[2] * tex_color[2])

    if shadowed:
        return vec3(ZERO, ZERO, ZERO)
    return cook_torrance_shading(n, v_vec, d_l, r_d, r_s, h_val, l_color)
