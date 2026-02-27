"""CUDA device functions for intersection tests."""

import math
from numba import cuda  # type: ignore

from utils.vec_utils import cross, dot, sub
from settings import CULLBACK, EPSILON, INFINITY_VEC


@cuda.jit(device=True)
def intersect_triangle(ro, rd, a, b, c):
    e1 = sub(b, a)
    e2 = sub(c, a)
    pvec = cross(rd, e2)
    det = dot(e1, pvec)

    if CULLBACK:
        if det < EPSILON:
            return INFINITY_VEC
    else:
        if math.fabs(det) < EPSILON:
            return INFINITY_VEC

    inv_det = 1.0 / det
    tvec = sub(ro, a)
    u = dot(tvec, pvec) * inv_det

    if u < 0.0 or u > 1.0:
        return INFINITY_VEC

    qvec = cross(tvec, e1)
    v = dot(rd, qvec) * inv_det

    if v < 0.0 or u + v > 1.0:
        return INFINITY_VEC

    t = dot(e2, qvec) * inv_det
    return t, u, v
