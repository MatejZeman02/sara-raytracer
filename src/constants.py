"""Constants for kernels."""

from numpy import float32

EPSILON = float32(1e-4)  # dynamic to size of the scene
BARYCENTRIC_EPSILON = float32(1e-7)  # acounts with float32 precision error
DENOMINATOR_EPSILON = float32(1e-10)  # division by zero and its check statements

THROUGHPUT_THRESHOLD = float32(1e-2)  # stop tracing rays where addition is negligible
INFINITY_VEC = (float32(-1.0), float32(0.0), float32(0.0))
BLOCK_THREADS = 16
STACK_SIZE = 64

# 32 bits float constants (It's almost like in OpenGl):
ZERO = float32(0.0)
HALF = float32(0.5)
ONE = float32(1.0)
TWO = float32(2.0)
