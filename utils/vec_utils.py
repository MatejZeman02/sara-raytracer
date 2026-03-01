"""Math vector utilities for CUDA kernels."""

import math
from numba import cuda


@cuda.jit(device=True, inline=True)
def vec3(x, y, z):
    """Create a 3D vector."""
    return (x, y, z)


@cuda.jit(device=True, inline=True)
def add(a, b):
    """+ operator for 3D vectors."""
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


@cuda.jit(device=True, inline=True)
def sub(a, b):
    """- operator for 3D vectors."""
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


@cuda.jit(device=True, inline=True)
def mul(a, s):
    """Scalar multiplication for 3D vectors."""
    return (a[0] * s, a[1] * s, a[2] * s)


@cuda.jit(device=True, inline=True)
def mul_vec(a, b):
    """Element-wise multiplication of two vectors."""
    return (a[0] * b[0], a[1] * b[1], a[2] * b[2])


@cuda.jit(device=True, inline=True)
def div(a, s):
    """Scalar division for 3D vectors."""
    return (a[0] / s, a[1] / s, a[2] / s)


@cuda.jit(device=True, inline=True)
def div_vec(a, b):
    """Element-wise division of two vectors."""
    return (a[0] / b[0], a[1] / b[1], a[2] / b[2])


@cuda.jit(device=True, inline=True)
def length(a):
    """Euclidean length of a 3D vector."""
    return math.sqrt(dot(a, a))


@cuda.jit(device=True, inline=True)
def dot(a, b):
    """Dot product for 3D vectors."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@cuda.jit(device=True, inline=True)
def normalize(a):
    """Normalize a 3D vector."""
    l = length(a)
    if l > 0:
        return mul(a, 1.0 / l)
    return a


@cuda.jit(device=True, inline=True)
def cross(a, b):
    """Cross product for 3D vectors."""
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


@cuda.jit(device=True, inline=True)
def reflect(v, n):
    """Reflect vector v around normal n."""
    # v - 2 * dot(v, n) * n
    d = dot(v, n)
    temp = mul(n, 2.0 * d)
    return sub(v, temp)


@cuda.jit(device=True, inline=True)
def distance(a, b):
    """Euclidean distance between two 3D points."""
    return length(sub(a, b))


@cuda.jit(device=True, inline=True)
def neg(a):
    """Negate a 3D vector."""
    return (-a[0], -a[1], -a[2])


@cuda.jit(device=True, inline=True)
def abs(a):
    """Element-wise absolute value of a vector."""
    return (abs(a[0]), abs(a[1]), abs(a[2]))


@cuda.jit(device=True, inline=True)
def clamp(a, min_val, max_val):
    """Element-wise clamp of a vector."""
    return (
        min(max(a[0], min_val), max_val),
        min(max(a[1], min_val), max_val),
        min(max(a[2], min_val), max_val),
    )


@cuda.jit(device=True, inline=True)
def mix(a, b, t):
    """Linear interpolation between vectors a and b by t."""
    return add(mul(a, 1.0 - t), mul(b, t))


@cuda.jit(device=True, inline=True)
def floor(a):
    """Element-wise floor of a vector."""
    return (math.floor(a[0]), math.floor(a[1]), math.floor(a[2]))


@cuda.jit(device=True, inline=True)
def ceil(a):
    """Element-wise ceil of a vector."""
    return (math.ceil(a[0]), math.ceil(a[1]), math.ceil(a[2]))


@cuda.jit(device=True, inline=True)
def min(a, b):
    """shorter vector between a and b from the origin."""
    return length(a) if length(a) < length(b) else length(b)


@cuda.jit(device=True, inline=True)
def max(a, b):
    """longer vector between a and b from the origin."""
    return length(a) if length(a) > length(b) else length(b)


@cuda.jit(device=True, inline=True)
def linear_to_srgb(c):
    """Convert linear color component (float) to sRGB."""
    if c <= 0.0031308:
        return 12.92 * c
    return 1.055 * math.pow(c, 1.0 / 2.4) - 0.055


@cuda.jit(device=True, inline=True)
def srgb_to_linear(c):
    """Convert sRGB color component (float) to linear."""
    if c <= 0.04045:
        return c / 12.92
    return math.pow((c + 0.055) / 1.055, 2.4)
