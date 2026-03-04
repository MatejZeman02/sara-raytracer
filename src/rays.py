import math
import numpy as np
from utils import device_jit
from utils.vec_utils import vec3, normalize, add, sub, mul, dot
from constants import DENOMINATOR_EPSILON, ZERO, ONE, EPSILON


@device_jit
def compute_inv_dir(dir_vec):
    """optimalization to divide only once per ray"""
    inv_x = ONE / (dir_vec[0] + DENOMINATOR_EPSILON)
    inv_y = ONE / (dir_vec[1] + DENOMINATOR_EPSILON)
    inv_z = ONE / (dir_vec[2] + DENOMINATOR_EPSILON)
    return vec3(inv_x, inv_y, inv_z)


@device_jit
def compute_primary_ray(p00, qw, qh, origin, x, y):
    """primary ray generation"""
    xf = np.float32(x)
    yf = np.float32(y)
    dir_x = p00[0] + xf * qw[0] - yf * qh[0]
    dir_y = p00[1] + xf * qw[1] - yf * qh[1]
    dir_z = p00[2] + xf * qw[2] - yf * qh[2]

    ray_dir = normalize(vec3(dir_x, dir_y, dir_z))
    ray_origin = vec3(origin[0], origin[1], origin[2])
    inv_rd = compute_inv_dir(ray_dir)
    return ray_origin, ray_dir, inv_rd


@device_jit
def compute_refraction(ray_dir, n, geom_n, p, ior, is_backface):
    """compute refraction ray direction and origin"""
    rel_ior = ior if is_backface else (ONE / (ior + DENOMINATOR_EPSILON))
    cos_i = dot(ray_dir, n)
    k = ONE - (rel_ior * rel_ior) * (ONE - (cos_i * cos_i))

    if k < ZERO:
        # total internal reflection
        new_dir = sub(ray_dir, mul(n, np.float32(2.0) * cos_i))
        new_origin = add(p, mul(geom_n, EPSILON))
    else:
        # pass through glass
        new_dir = normalize(
            add(
                mul(ray_dir, rel_ior),
                mul(n, rel_ior * (-cos_i) - np.float32(math.sqrt(k))),
            )
        )
        # step through the opposing geometry wall
        new_origin = sub(p, mul(geom_n, EPSILON))
    return new_dir, new_origin


@device_jit
def compute_reflection(ray_dir, n, geom_n, p):
    """compute reflection ray direction and origin"""
    cos_i = dot(ray_dir, n)
    new_dir = sub(ray_dir, mul(n, np.float32(2.0) * cos_i))
    new_origin = add(p, mul(geom_n, EPSILON))
    return new_dir, new_origin
