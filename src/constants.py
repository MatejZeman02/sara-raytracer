"""Constants for kernels."""

import math
from numpy import float32, uint8

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
UINT8_MAX_F = float32(255.0)
UINT8_MAX_I = uint8(255)
PI = float32(math.pi)

# out stats layout:
# 0: primary tri tests
PRIMARY_TRI = 0
# 1: primary node tests
PRIMARY_NODE = 1
# 2: primary rays
PRIMARY_RAY = 2
# 3: secondary rays (refractions + bounces)
SECONDARY_RAY = 3
# 4: shadow rays
SHADOW_RAY = 4
