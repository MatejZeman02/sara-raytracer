"""Device functions for intersection tests."""

import math
from numpy import float32

from utils import device_jit
from utils.vec_utils import vec3, mul_vec, cross, dot, sub
from settings import CULLBACK
from constants import EPSILON, INFINITY_VEC

@device_jit
def intersect_aabb(ro, inv_rd, bmin, bmax):
    """Ray-AABB intersection test using the slab method."""
    # calculate intersection t-values for all axes
    t1 = mul_vec(sub(bmin, ro), inv_rd)
    t2 = mul_vec(sub(bmax, ro), inv_rd)

    # find min and max for x, y, z
    tmin_x = min(t1[0], t2[0])
    tmax_x = max(t1[0], t2[0])

    tmin_y = min(t1[1], t2[1])
    tmax_y = max(t1[1], t2[1])

    tmin_z = min(t1[2], t2[2])
    tmax_z = max(t1[2], t2[2])

    # slab method overlap check
    tmin = max(tmin_x, max(tmin_y, tmin_z))
    tmax = min(tmax_x, min(tmax_y, tmax_z))

    # ray hits box if tmax is greater than or equal to max(tmin, 0.0)
    hit = tmax >= max(tmin, float32(0.0))

    return hit, tmin


@device_jit
def intersect_triangle(ro, rd, a, b, c):
    """Möller-Trumbore ray-triangle intersection algorithm."""
    e1 = sub(b, a)
    e2 = sub(c, a)
    pvec = cross(rd, e2)
    det = dot(e1, pvec)

    if CULLBACK:
        if det < EPSILON:
            return INFINITY_VEC
    else:
        if float32(math.fabs(det)) < EPSILON:
            return INFINITY_VEC

    inv_det = float32(1.0) / det
    tvec = sub(ro, a)
    u = dot(tvec, pvec) * inv_det

    if u < float32(0.0) or u > float32(1.0):
        return INFINITY_VEC

    qvec = cross(tvec, e1)
    v = dot(rd, qvec) * inv_det

    if v < float32(0.0) or u + v > float32(1.0):
        return INFINITY_VEC

    t = dot(e2, qvec) * inv_det
    return t, u, v
