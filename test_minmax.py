import numpy as np
import numba

@numba.njit(fastmath=True)
def f_test():
    a = np.float32(-1e10)
    b = np.float32(1e10)
    return min(a, b), max(a, b)

print(f_test())
