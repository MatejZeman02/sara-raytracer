import numpy as np
from numba import njit
from numpy import uint32, float32

def rand_float32_py(rng_states, idx):
    x = uint32(rng_states[idx])
    x = uint32(x ^ uint32(x << uint32(13)))
    x = uint32(x ^ uint32(x >> uint32(17)))
    x = uint32(x ^ uint32(x << uint32(5)))
    rng_states[idx] = x
    return float32(x) * float32(2.3283064365386963e-10)

@njit
def rand_float32_jit(rng_states, idx):
    x = rng_states[idx]
    x = x ^ (x << uint32(13))
    x = x ^ (x >> uint32(17))
    x = x ^ (x << uint32(5))
    rng_states[idx] = x
    return float32(x) * float32(2.3283064365386963e-10)

states_py = np.array([42], dtype=uint32)
states_jit = np.array([42], dtype=uint32)

print("PY:", rand_float32_py(states_py, 0), rand_float32_py(states_py, 0))
print("JIT:", rand_float32_jit(states_jit, 0), rand_float32_jit(states_jit, 0))
