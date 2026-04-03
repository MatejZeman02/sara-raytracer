import numpy as np
import numba

@numba.njit
def intersect(ro, inv_rd, bmin, bmax):
    t1_x = (bmin[0] - ro[0]) * inv_rd[0]
    t2_x = (bmax[0] - ro[0]) * inv_rd[0]
    if np.isnan(t1_x) or np.isnan(t2_x):
        return True
    return False

def test():
    pass

