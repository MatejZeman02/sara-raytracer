import numpy as np
import numba

@numba.njit(fastmath=True)
def get_inv_rd(d):
    eps = np.float32(1e-10)
    return (np.float32(1.0)/(d[0]+eps), np.float32(1.0)/(d[1]+eps), np.float32(1.0)/(d[2]+eps))

@numba.njit(fastmath=True)
def intersect_numba(ro, inv_rd, bmin, bmax):
    t1 = (bmin - ro) * inv_rd
    t2 = (bmax - ro) * inv_rd
    tmin_x = min(t1[0], t2[0]); tmax_x = max(t1[0], t2[0])
    tmin_y = min(t1[1], t2[1]); tmax_y = max(t1[1], t2[1])
    tmin_z = min(t1[2], t2[2]); tmax_z = max(t1[2], t2[2])
    tmin = max(tmin_x, max(tmin_y, tmin_z))
    tmax = min(tmax_x, min(tmax_y, tmax_z))
    return tmax >= max(tmin, np.float32(0.0)), tmin

def intersect_py(ro, inv_rd, bmin, bmax):
    t1 = (bmin - ro) * inv_rd
    t2 = (bmax - ro) * inv_rd
    tmin_x = min(t1[0], t2[0]); tmax_x = max(t1[0], t2[0])
    tmin_y = min(t1[1], t2[1]); tmax_y = max(t1[1], t2[1])
    tmin_z = min(t1[2], t2[2]); tmax_z = max(t1[2], t2[2])
    tmin = max(tmin_x, max(tmin_y, tmin_z))
    tmax = min(tmax_x, min(tmax_y, tmax_z))
    return tmax >= max(tmin, np.float32(0.0)), tmin

np.random.seed(0)
for i in range(100):
    ro = np.array([-1, 0.5, 0], dtype=np.float32)
    d = np.array([0, 1.0, 1.0], dtype=np.float32)
    inv_rd = get_inv_rd(d)
    bmin = np.array([-1, -1, -1], dtype=np.float32)
    bmax = np.array([1, 1, 1], dtype=np.float32)
    res_num = intersect_numba(ro, np.array(inv_rd, dtype=np.float32), bmin, bmax)
    res_py = intersect_py(ro, np.array(inv_rd, dtype=np.float32), bmin, bmax)
    if res_num != res_py:
        print("Diff at", i, ro, d, inv_rd, bmin, bmax)
        print("numba", res_num, "py", res_py)
        break

