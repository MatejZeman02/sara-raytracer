import math
from numpy import float32
from utils import device_jit
from utils.vec_utils import vec3, normalize, add, sub, mul, dot
from .constants import DENOMINATOR_EPSILON, TWO, ZERO, ONE, RAY_EPSILON


@device_jit
def compute_inv_dir(dir_vec):
    """optimalization to divide only once per ray"""
    inv_x = ONE / (dir_vec[0] + DENOMINATOR_EPSILON)
    inv_y = ONE / (dir_vec[1] + DENOMINATOR_EPSILON)
    inv_z = ONE / (dir_vec[2] + DENOMINATOR_EPSILON)

    return vec3(inv_x, inv_y, inv_z)


@device_jit
def compute_primary_ray(p00, qw, qh, origin, x, y, jx, jy):
    """primary ray generation with optional sub-pixel jitter for anti-aliasing"""
    # float conversion needed because of the multiplication; jx/jy offset in [-0.5, 0.5)
    xf = float32(x) + jx  # pixel_x_with_jitter
    yf = float32(y) + jy  # pixel_y_with_jitter
    dir_x = p00[0] + xf * qw[0] - yf * qh[0]
    dir_y = p00[1] + xf * qw[1] - yf * qh[1]
    dir_z = p00[2] + xf * qw[2] - yf * qh[2]

    ray_dir = normalize(vec3(dir_x, dir_y, dir_z))
    ray_origin = vec3(origin[0], origin[1], origin[2])
    inv_rd = compute_inv_dir(ray_dir)  # dir^-1
    # check for NaN
    assert ray_dir[0] == ray_dir[0]
    assert ray_dir[1] == ray_dir[1]
    assert ray_dir[2] == ray_dir[2]

    return ray_origin, ray_dir, inv_rd


@device_jit
def compute_refraction(ray_dir, n, geom_n, p, ior, is_backface):
    """compute refraction ray direction and origin"""
    # relative index of refraction depends on if ray is entering or exiting
    rel_ior = ior if is_backface else (ONE / (ior + DENOMINATOR_EPSILON))
    # incidence angle
    cos_i = dot(ray_dir, n)  # cosine_incidence
    # Snell's law discriminant 'k'
    k = ONE - (rel_ior * rel_ior) * (ONE - (cos_i * cos_i))  # snell_discriminant

    if k < ZERO:
        # total internal reflection
        new_dir = sub(ray_dir, mul(n, TWO * cos_i))
        # shift forwards
        new_origin = add(p, mul(geom_n, RAY_EPSILON))
    else:
        # pass through glass
        new_dir = normalize(
            add(
                mul(ray_dir, rel_ior),
                mul(n, rel_ior * (-cos_i) - math.sqrt(k)),
            )
        )
        # shift forwards
        new_origin = sub(p, mul(geom_n, RAY_EPSILON))
    return new_dir, new_origin


@device_jit
def compute_reflection(ray_dir, n, geom_n, p):
    """compute reflection ray direction and origin"""
    # incidence angle
    cos_i = dot(ray_dir, n)  # cosine_incidence
    new_dir = sub(ray_dir, mul(n, TWO * cos_i))
    # shift backwards
    new_origin = add(p, mul(geom_n, RAY_EPSILON))
    return new_dir, new_origin
