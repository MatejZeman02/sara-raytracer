"""area light sampling for next-event estimation (soft shadows)."""

from math import sqrt
from numpy import float32, int32

from utils import device_jit
from utils.vec_utils import vec3, sub, cross, dot, normalize, mul
from .geometry import get_tri_verts
from .constants import (
    TWO,
    ZERO,
    HALF,
    ONE,
    RAY_EPSILON,
    MAT_EMISSIVE_R,
    MAT_EMISSIVE_G,
    MAT_EMISSIVE_B,
)
from .rng import rand_float32


@device_jit
def triangle_area(a, b, c):
    """compute the area of a triangle from its three vertices."""
    e1 = sub(b, a)
    e2 = sub(c, a)
    cr = cross(e1, e2)
    return HALF * sqrt(cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2])


@device_jit
def triangle_face_normal(a, b, c):
    """compute the unit-length outward face normal of a triangle."""
    e1 = sub(b, a)
    e2 = sub(c, a)
    cr = cross(e1, e2)
    len_cr = sqrt(cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2])
    if len_cr > ZERO:
        inv = ONE / len_cr
        return vec3(cr[0] * inv, cr[1] * inv, cr[2] * inv)
    # fallback
    return vec3(ZERO, ONE, ZERO)


@device_jit
def sample_triangle_point(a, b, c, r1, r2):
    """uniformly sample a point on a triangle - (Osada et al. 2002 mapping)"""
    assert r1 >= ZERO
    assert r2 >= ZERO
    # square-root warping maps the unit square to uniform barycentric coordinates
    sqrt_r1 = sqrt(r1)
    u = ONE - sqrt_r1  # barycentric_u
    v = r2 * sqrt_r1  # barycentric_v
    w = ONE - u - v  # barycentric_w
    return vec3(
        u * a[0] + v * b[0] + w * c[0],
        u * a[1] + v * b[1] + w * c[1],
        u * a[2] + v * b[2] + w * c[2],
    )


@device_jit
def sample_area_light(
    triangles,
    mat_indices,
    materials,
    emissive_tris,
    num_emissive,
    p,
    rng_states,
    thread_idx,
):
    """
    uniformly select one emissive triangle and importance-sample a point on it.

    returns: weighted_emission already encodes full area-to-solid-angle
    conversion and discrete selection pdf:
        weighted_emission = emission * cos_theta_light * num_emissive * area / dist^2

    degenerate cases (zero area, backfacing sample, coincident point) return zero
    weighted_emission so can unconditionally evaluate brdf.
    """
    assert num_emissive > 0

    # draw three independent uniform floats: one for triangle selection, two for uv
    r0 = rand_float32(rng_states, thread_idx)  # random_select
    r1 = rand_float32(rng_states, thread_idx)  # random_bary_u
    r2 = rand_float32(rng_states, thread_idx)  # random_bary_v

    # discrete uniform selection over all emissive triangles
    tri_sel = int32(r0 * float32(num_emissive))
    if tri_sel >= num_emissive:
        tri_sel = num_emissive - 1  # r0 is near 1.0
    light_tri_idx = emissive_tris[tri_sel]
    a, b, c = get_tri_verts(triangles, light_tri_idx)

    # uniformly sample a point on the triangle surface
    l_pos = sample_triangle_point(a, b, c, r1, r2)

    # (unnormalized) direction from shading point to the light sample
    l_dir_vec = sub(l_pos, p)
    dist_sq = dot(l_dir_vec, l_dir_vec)
    dist_to_light = float32(sqrt(dist_sq))

    if dist_to_light < RAY_EPSILON:
        # shading point is essentially on the light surface - skip contribution
        return (
            vec3(ZERO, ONE, ZERO),
            float32(RAY_EPSILON * TWO),
            vec3(ZERO, ZERO, ZERO),
        )

    d_l = normalize(l_dir_vec)  # direction_to_light

    # outward face normal of the emitter
    light_n = triangle_face_normal(a, b, c)

    # cosine at the emitter surface: must be positive for a front-facing sample
    # dot(light_n, -d_l) measures alignment between emitter normal and ray back to surface
    cos_theta_light = dot(light_n, vec3(-d_l[0], -d_l[1], -d_l[2]))

    if cos_theta_light <= ZERO:
        # sample landed on the back side of the emitter - no emission in this direction
        return d_l, dist_to_light, vec3(ZERO, ZERO, ZERO)

    # fetch emission radiance from the triangle's material (columns 7, 8, 9)
    mat_idx = mat_indices[light_tri_idx]
    emission = vec3(
        materials[mat_idx, MAT_EMISSIVE_R],
        materials[mat_idx, MAT_EMISSIVE_G],
        materials[mat_idx, MAT_EMISSIVE_B],
    )

    area = triangle_area(a, b, c)
    if area < RAY_EPSILON:
        # degenerate zero-area triangle - cannot contribute
        return d_l, dist_to_light, vec3(ZERO, ZERO, ZERO)

    # monte carlo weight for next-event estimation:
    #   pdf  = 1 / (num_emissive * area)   [uniform discrete + uniform area sampling]
    #   G    = cos_theta_light / dist_sq
    #   weight = G / pdf = cos_theta_light * num_emissive * area / dist_sq
    weight = cos_theta_light * float32(num_emissive) * area / dist_sq

    weighted_emission = vec3(
        emission[0] * weight,
        emission[1] * weight,
        emission[2] * weight,
    )
    return d_l, dist_to_light, weighted_emission
