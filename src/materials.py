import math
import numpy as np
from utils import device_jit
from utils.vec_utils import vec3, add, sub, mul, neg, dot
from shading import cook_torrance_shading
from traversal import is_in_shadow
from rays import compute_inv_dir
from constants import ZERO, EPSILON


@device_jit
def get_emissive_color(materials, mat_idx):
    return vec3(materials[mat_idx, 7], materials[mat_idx, 8], materials[mat_idx, 9])


@device_jit
def compute_shadow_ray_origin(
    a, b, c, na, nb, nc, w, hit_u, hit_v, p, geom_n, is_backface
):
    """CHIANG'S analytical terminator offset @Disney 2019"""
    if na[0] == ZERO and na[1] == ZERO and na[2] == ZERO:
        return add(p, mul(geom_n, EPSILON))

    fa_ = neg(na) if is_backface else na
    fb_ = neg(nb) if is_backface else nb
    fc_ = neg(nc) if is_backface else nc

    p_a = add(p, mul(fa_, dot(sub(a, p), fa_)))
    p_b = add(p, mul(fb_, dot(sub(b, p), fb_)))
    p_c = add(p, mul(fc_, dot(sub(c, p), fc_)))

    p_curve = add(add(mul(p_a, w), mul(p_b, hit_u)), mul(p_c, hit_v))
    return add(p_curve, mul(geom_n, EPSILON))


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
    return is_in_shadow(
        triangles, bvh_nodes, use_bvh, shadow_ro, d_l, inv_dl, dist_to_light, stack
    )


@device_jit
def compute_lit_color(materials, mat_idx, light_color, n, v_vec, d_l, shadowed):
    """Shading"""
    # fetch material properties
    r_d = vec3(materials[mat_idx, 0], materials[mat_idx, 1], materials[mat_idx, 2])
    r_s = vec3(materials[mat_idx, 3], materials[mat_idx, 4], materials[mat_idx, 5])
    h_val = materials[mat_idx, 6]
    l_color = vec3(light_color[0], light_color[1], light_color[2])

    if shadowed:
        return vec3(ZERO, ZERO, ZERO)
    return cook_torrance_shading(n, v_vec, d_l, r_d, r_s, h_val, l_color)
