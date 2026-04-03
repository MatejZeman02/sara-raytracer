import numpy as np
import numba

@numba.njit
def sub(a, b): return (a[0]-b[0], a[1]-b[1], a[2]-b[2])
@numba.njit
def cross(a, b): return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
@numba.njit
def dot(a, b): return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]

@numba.njit(fastmath=True)
def intersect_numba(ro, rd, a, b, c, cullback):
    e1 = sub(b, a); e2 = sub(c, a); pvec = cross(rd, e2); det = dot(e1, pvec)
    if cullback:
        if det <= np.float32(1e-9): return False, 0.0
    else:
        if abs(det) < np.float32(1e-9): return False, 0.0
    inv_det = np.float32(1.0) / det; tvec = sub(ro, a); u = dot(tvec, pvec) * inv_det
    if u < 0.0 or u > 1.0: return False, 0.0
    qvec = cross(tvec, e1); v = dot(rd, qvec) * inv_det
    if v < 0.0 or u + v > 1.0: return False, 0.0
    return True, dot(e2, qvec) * inv_det

def intersect_py(ro, rd, a, b, c, cullback):
    e1 = sub(b, a); e2 = sub(c, a); pvec = cross(rd, e2); det = dot(e1, pvec)
    if cullback:
        if det <= np.float32(1e-9): return False, 0.0
    else:
        if abs(det) < np.float32(1e-9): return False, 0.0
    inv_det = np.float32(1.0) / det; tvec = sub(ro, a); u = dot(tvec, pvec) * inv_det
    if u < 0.0 or u > 1.0: return False, 0.0
    qvec = cross(tvec, e1); v = dot(rd, qvec) * inv_det
    if v < 0.0 or u + v > 1.0: return False, 0.0
    return True, dot(e2, qvec) * inv_det

np.random.seed(42)
for i in range(1000):
    ro = np.random.randn(3).astype(np.float32)
    rd = np.random.randn(3).astype(np.float32)
    a = np.random.randn(3).astype(np.float32)
    b = np.random.randn(3).astype(np.float32)
    c = np.random.randn(3).astype(np.float32)
    n = intersect_numba(ro, rd, a, b, c, True)
    p = intersect_py(ro, rd, a, b, c, True)
    if n != p:
        print("Diff!", n, p, ro, rd, a, b, c)
