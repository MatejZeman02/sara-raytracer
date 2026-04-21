"""per-pixel pseudo-random number generator - device-compatible on both cpu and gpu."""

import numpy as np
from numpy import uint32, float32

from numba.cuda.random import (
    xoroshiro128p_uniform_float32,
    create_xoroshiro128p_states,
)

from settings import DEVICE
from utils import device_jit

if DEVICE == "cpu":

    @device_jit
    def rand_float32(rng_states, idx):
        """xorshift32 prng: advance state in-place and return a uniform float in [0, 1)."""
        x = uint32(rng_states[idx])
        x ^= uint32((x << uint32(13)) & uint32(0xFFFFFFFF))
        x ^= uint32((x >> uint32(17)) & uint32(0xFFFFFFFF))
        x ^= uint32((x << uint32(5)) & uint32(0xFFFFFFFF))
        rng_states[idx] = x
        return float32(x) * float32(2.3283064365386963e-10)

    def create_rng_states(n: int, seed: int = 42):
        """allocate a uint32 state array seeded uniquely per pixel index."""
        n_i32 = np.int32(n)
        seed_u32 = uint32(seed)
        assert n_i32 > 0
        # seed each entry with its index + 1 to avoid zero state (xorshift dies at 0)
        states = np.arange(1, int(n_i32) + 1, dtype=uint32)
        # override entry 0 with the caller-provided seed
        states[0] = max(uint32(1), seed_u32)
        return states

else:

    @device_jit
    def rand_float32(rng_states, idx):
        """advance xoroshiro128p state and return a uniform float in [0, 1)."""
        return xoroshiro128p_uniform_float32(rng_states, idx)

    def create_rng_states(n: int, seed: int = 42):
        """allocate xoroshiro128p states on the gpu - returns a device array."""
        n_i32 = np.int32(n)
        seed_u64 = np.uint64(seed)
        assert n_i32 > 0
        return create_xoroshiro128p_states(int(n_i32), seed=int(seed_u64))
