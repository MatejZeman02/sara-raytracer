"""Device functions for intersection tests."""

import math

from utils import device_jit
from utils.vec_utils import vec3, mul_vec, cross, dot, sub
from constants import DET_EPSILON, INFINITY_VEC, ZERO, ONE


@device_jit
def intersect_aabb(ro, inv_rd, bmin, bmax):
    """Ray-AABB intersection test using the slab method."""
    # calculate intersection t-values for all axes
    assert inv_rd[0] != 0.0
    assert inv_rd[1] != 0.0
    assert inv_rd[2] != 0.0
    # check for NaN
    assert inv_rd[0] == inv_rd[0]
    assert inv_rd[1] == inv_rd[1]
    assert inv_rd[2] == inv_rd[2]
    t1 = mul_vec(sub(bmin, ro), inv_rd)  # near_t_per_axis
    t2 = mul_vec(sub(bmax, ro), inv_rd)  # far_t_per_axis

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
    hit = tmax >= max(tmin, ZERO)

    return hit, tmin


@device_jit
def intersect_triangle(ro, rd, a, b, c, cullback):
    """Möller-Trumbore ray-triangle intersection algorithm."""
    e1 = sub(b, a)  # edge_1
    e2 = sub(c, a)  # edge_2
    pvec = cross(rd, e2)
    det = dot(e1, pvec)

    # primary rays cull backfaces to get into the scene
    if cullback:
        if det < DET_EPSILON:
            return INFINITY_VEC
    else:
        if math.fabs(det) < DET_EPSILON:
            return INFINITY_VEC

    inv_det = ONE / det
    tvec = sub(ro, a)
    u = dot(tvec, pvec) * inv_det  # barycentric_u

    if u < ZERO or u > ONE:
        return INFINITY_VEC

    qvec = cross(tvec, e1)
    v = dot(rd, qvec) * inv_det  # barycentric_v

    if v < ZERO or u + v > ONE:
        return INFINITY_VEC

    t = dot(e2, qvec) * inv_det  # hit_distance
    return t, u, v
