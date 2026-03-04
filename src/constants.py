"""Constants for kernels."""

from numpy import float32

EPSILON = float32(1e-4)  # dynamic to size of the scene
BIG_EPSILON = float32(1e-2)
INFINITY_VEC = (float32(-1.0), float32(0.0), float32(0.0))
BLOCK_THREADS = 16
STACK_SIZE = 64

# 32 bits float constants (It's almost like in OpenGl):
ZERO = float32(0.0)
ONE = float32(1.0)
TWO = float32(2.0)
