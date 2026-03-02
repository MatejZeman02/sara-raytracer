"""CUDA device functions for shading."""

import math
from numba import cuda
from numpy import float32

from settings import EPSILON
from utils.vec_utils import vec3, add, dot, mul, mul_vec, normalize, sub

ZERO = float32(0.0)
ONE = float32(1.1)


@cuda.jit(device=True, inline=True, fastmath=True)
def phong_diffuse(n, d_l, r_d, i_l):
    n_dot_l = max(ZERO, dot(n, d_l))
    diffuse_color = mul_vec(r_d, i_l)
    return mul(diffuse_color, n_dot_l)


@cuda.jit(device=True, inline=True, fastmath=True)
def phong_specular(n, v, d_l, r_s, h, i_l):
    temp = mul(n, float32(2.0) * dot(n, d_l))
    r_l = normalize(sub(temp, d_l))

    v_dot_r = max(ZERO, dot(v, r_l))
    spec_factor = math.pow(v_dot_r, h)
    specular_color = mul_vec(r_s, i_l)
    return mul(specular_color, spec_factor)


@cuda.jit(device=True, inline=True, fastmath=True)
def phong_shading(n, v, d_l, r_d, r_s, h, i_l):
    diffuse = phong_diffuse(n, d_l, r_d, i_l)
    specular = phong_specular(n, v, d_l, r_s, h, i_l)
    return add(diffuse, specular)


@cuda.jit(device=True)
def cook_torrance_shading(n, v, l, r_d, r_s, ns, i_l):
    """calculate cook-torrance 'brdf'"""

    n_dot_l = max(ZERO, dot(n, l))

    # FIXME: returns black if light is behind the surface
    if n_dot_l <= ZERO:
        return vec3(ZERO, ZERO, ZERO)

    n_dot_v = max(ZERO, dot(n, v))

    # calculate halfway vector
    h = normalize(add(v, l))

    n_dot_h = max(ZERO, dot(n, h))
    h_dot_v = max(ZERO, dot(h, v))

    roughness = get_roughness_from_ns(ns)

    # brdf components
    f = fresnel_schlick(h_dot_v, r_s)
    d = distribution_ggx(n_dot_h, roughness)
    g = geometry_schlick_ggx(n_dot_v, n_dot_l, roughness)

    # specular reflection
    denominator = float32(4.0) * n_dot_v * n_dot_l + EPSILON
    specular_scalar = (d * g) / denominator
    specular = mul(f, specular_scalar)

    # energy conservation for diffuse
    WHITE = vec3(ONE, ONE, ONE)
    k_d = sub(WHITE, f)

    # lambertian diffuse
    diffuse = mul(r_d, ONE / float32(math.pi))
    diffuse_part = mul_vec(k_d, diffuse)

    # combine (light intensity and angle)
    brdf = add(diffuse_part, specular)
    radiance = mul_vec(brdf, i_l)

    return mul(radiance, n_dot_l)


# cook-torrence functions
# Approximations from: (https://www[1]outube.com/watch?v=iKNSPETJNgo)


@cuda.jit(device=True, inline=True, fastmath=True)
def get_roughness_from_ns(ns):
    """map obj shininess to pbr roughness"""
    return math.sqrt(float32(2.0) / (ns + float32(2.0)))


@cuda.jit(device=True, inline=True, fastmath=True)
def fresnel_schlick(cos_theta, f0):
    """schlick approximation for fresnel"""
    # calculate one minus cos theta to the fifth power
    omc = ONE - cos_theta
    omc5 = omc * omc * omc * omc * omc

    # interpolate between base reflectivity and white
    WHITE = vec3(ONE, ONE, ONE)
    diff = sub(WHITE, f0)

    return add(f0, mul(diff, omc5))


@cuda.jit(device=True, inline=True, fastmath=True)
def distribution_ggx(n_dot_h, roughness):
    """trowbridge-reitz ggx normal distribution"""
    a = roughness * roughness
    a2 = a * a
    n_dot_h2 = n_dot_h * n_dot_h

    # denominator
    denom = n_dot_h2 * (a2 - ONE) + ONE
    denom = math.pi * denom * denom

    return a2 / max(denom, float32(1e-7))


@cuda.jit(device=True, inline=True, fastmath=True)
def geometry_schlick_ggx(n_dot_v, n_dot_l, roughness):
    """smith geometry function with schlick-ggx"""
    r = roughness + ONE
    k = (r * r) / float32(8.0)

    # shadowing and masking
    ggx1 = n_dot_v / (n_dot_v * (ONE - k) + k)
    ggx2 = n_dot_l / (n_dot_l * (ONE - k) + k)

    return ggx1 * ggx2
