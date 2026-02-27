"""CUDA device functions for shading."""

import math
from numba import cuda  # type: ignore

from utils.vec_utils import add, div_vec, dot, mul, mul_vec, normalize, sub, vec3


@cuda.jit(device=True)
def phong_diffuse(n, d_l, r_d, i_l):
    n_dot_l = max(0.0, dot(n, d_l))
    diffuse_color = mul_vec(r_d, i_l)
    return mul(diffuse_color, n_dot_l)


@cuda.jit(device=True)
def phong_specular(n, v, d_l, r_s, h, i_l):
    temp = mul(n, 2.0 * dot(n, d_l))
    r_l = normalize(sub(temp, d_l))

    v_dot_r = max(0.0, dot(v, r_l))
    spec_factor = math.pow(v_dot_r, h)
    specular_color = mul_vec(r_s, i_l)
    return mul(specular_color, spec_factor)


@cuda.jit(device=True)
def phong_shading(n, v, d_l, r_d, r_s, h, i_l):
    diffuse = phong_diffuse(n, d_l, r_d, i_l)
    specular = phong_specular(n, v, d_l, r_s, h, i_l)
    return add(diffuse, specular)


# @cuda.jit(device=True)
# def coock_torrence(n, v, d_l, r_d, r_s, h, i_l):
#     n_dot_l = max(0.0, dot(n, d_l))
#     n_dot_v = max(0.0, dot(n, v))
#     n_dot_h = max(0.0, dot(n, h))
#     v_dot_h = max(0.0, dot(v, h))

#     f0 = r_s
#     d = distribution_ggx(n_dot_h, h)  # Normal Distribution Function approximation
#     f = fresnel_schlick(v_dot_h, f0)  # Fresnel approximation
#     g = geometry_schlick_ggx(n_dot_v, n_dot_l, h)  # Geometry function approximation

#     numerator = mul_vec(f, mul_vec(vec3(d * g, d * g, d * g), i_l))
#     denominator = 4.0 * n_dot_l * n_dot_v + 1e-7  # avoiding division by zero
#     specular = div_vec(numerator, vec3(denominator, denominator, denominator))

#     # diffuse component with energy conservation
#     k_d = sub(vec3(1.0, 1.0, 1.0), f)  # energy conservation factor
#     diffuse = mul_vec(mul_vec(r_d, k_d), i_l)

#     return add(diffuse, specular)


@cuda.jit(device=True)
def cook_torrance_shading(n, v, l, r_d, r_s, ns, i_l):
    """calculate cook-torrance 'brdf'"""

    n_dot_l = max(0.0, dot(n, l))

    # FIXME: returns black if light is behind the surface
    if n_dot_l <= 0.0:
        return vec3(0.0, 0.0, 0.0)

    n_dot_v = max(0.0, dot(n, v))

    # calculate halfway vector
    h = normalize(add(v, l))

    n_dot_h = max(0.0, dot(n, h))
    h_dot_v = max(0.0, dot(h, v))

    roughness = get_roughness_from_ns(ns)

    # brdf components
    f = fresnel_schlick(h_dot_v, r_s)
    d = distribution_ggx(n_dot_h, roughness)
    g = geometry_schlick_ggx(n_dot_v, n_dot_l, roughness)

    # specular reflection
    denominator = 4.0 * n_dot_v * n_dot_l + 0.0001
    specular_scalar = (d * g) / denominator
    specular = mul(f, specular_scalar)

    # energy conservation for diffuse
    white = vec3(1.0, 1.0, 1.0)
    k_d = sub(white, f)

    # lambertian diffuse
    diffuse = mul(r_d, 1.0 / math.pi)
    diffuse_part = mul_vec(k_d, diffuse)

    # combine (light intensity and angle)
    brdf = add(diffuse_part, specular)
    radiance = mul_vec(brdf, i_l)

    return mul(radiance, n_dot_l)


# coock-torrence functions
# Approximations from: (https://www.youtube.com/watch?v=iKNSPETJNgo)


@cuda.jit(device=True, inline=True)
def get_roughness_from_ns(ns):
    """map obj shininess to pbr roughness"""
    return math.sqrt(2.0 / (ns + 2.0))


@cuda.jit(device=True, inline=True)
def fresnel_schlick(cos_theta, f0):
    """schlick approximation for fresnel"""
    # calculate one minus cos theta to the fifth power
    omc = 1.0 - cos_theta
    omc5 = omc * omc * omc * omc * omc

    # interpolate between base reflectivity and white
    white = vec3(1.0, 1.0, 1.0)
    diff = sub(white, f0)

    return add(f0, mul(diff, omc5))


@cuda.jit(device=True, inline=True)
def distribution_ggx(n_dot_h, roughness):
    """trowbridge-reitz ggx normal distribution"""
    a = roughness * roughness
    a2 = a * a
    n_dot_h2 = n_dot_h * n_dot_h

    # denominator
    denom = n_dot_h2 * (a2 - 1.0) + 1.0
    denom = math.pi * denom * denom

    return a2 / max(denom, 1e-7)


@cuda.jit(device=True, inline=True)
def geometry_schlick_ggx(n_dot_v, n_dot_l, roughness):
    """smith geometry function with schlick-ggx"""
    r = roughness + 1.0
    k = (r * r) / 8.0

    # shadowing and masking
    ggx1 = n_dot_v / (n_dot_v * (1.0 - k) + k)
    ggx2 = n_dot_l / (n_dot_l * (1.0 - k) + k)

    return ggx1 * ggx2
