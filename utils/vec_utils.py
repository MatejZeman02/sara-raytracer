"""Math vector utilities for device-agnostic kernels. Commented out functions are not currently used."""

import math
from numpy import float32

from utils import device_jit

# constants for float32 precision
ONE = float32(1.0)
ZERO = float32(0.0)


@device_jit
def vec3(x, y, z):
    """Create a 3D vector."""
    return (x, y, z)


@device_jit
def add(a, b):
    """+ operator for 3D vectors."""
    a = vec3(a[0], a[1], a[2])
    b = vec3(b[0], b[1], b[2])
    return (a[0] + b[0]), (a[1] + b[1]), (a[2] + b[2])


@device_jit
def sub(a, b):
    """- operator for 3D vectors."""
    a = vec3(a[0], a[1], a[2])
    b = vec3(b[0], b[1], b[2])
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


@device_jit
def mul(a, s):
    """Scalar multiplication for 3D vectors."""
    a = vec3(a[0], a[1], a[2])
    return (a[0] * s, a[1] * s, a[2] * s)


@device_jit
def mul_vec(a, b):
    """Element-wise multiplication of two vectors."""
    return (
        a[0] * b[0],
        a[1] * b[1],
        a[2] * b[2],
    )


# @device_jit
# def div(a, s):
#     """Scalar division for 3D vectors."""
#     return float32((a[0] / s, a[1] / s, a[2] / s))


# @device_jit
# def div_vec(a, b):
#     """Element-wise division of two vectors."""
#     return (float32(a[0] / b[0]), float32(a[1] / b[1]), float32(a[2] / b[2]))


@device_jit
def length(a):
    """Euclidean length of a 3D vector."""
    return math.sqrt(dot(a, a))


@device_jit
def dot(a, b):
    """Dot product for 3D vectors."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@device_jit
def normalize(a):
    """Normalize a 3D vector."""
    l = length(a)
    if l > ZERO:
        return mul(a, ONE / l)
    return vec3(a[0], a[1], a[2])


@device_jit
def cross(a, b):
    """Cross product for 3D vectors."""
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


# @cuda.jit(device=True, inline=True, fastmath=False)
# def distance(a, b):
#     """Euclidean distance between two 3D points."""
#     return length(sub(a, b))


@device_jit
def neg(a):
    """Negate a 3D vector."""
    return (float32(-a[0]), float32(-a[1]), float32(-a[2]))


# @cuda.jit(device=True, inline=True, fastmath=False)
# def abs(a):
#     """Element-wise absolute value of a vector."""
#     return (abs(a[0]), abs(a[1]), abs(a[2]))


# @cuda.jit(device=True, inline=True, fastmath=False)
# def clamp(a, min_val, max_val):
#     """Element-wise clamp of a vector."""
#     return (
#         min(max(a[0], min_val), max_val),
#         min(max(a[1], min_val), max_val),
#         min(max(a[2], min_val), max_val),
#     )


# @cuda.jit(device=True, inline=True, fastmath=False)
# def mix(a, b, t):
#     """Linear interpolation between vectors a and b by t."""
#     return add(mul(a, ONE - t), mul(b, t))


# @cuda.jit(device=True, inline=True, fastmath=False)
# def floor(a):
#     """Element-wise floor of a vector."""
#     return (math.floor(a[0]), math.floor(a[1]), math.floor(a[2]))


# @cuda.jit(device=True, inline=True, fastmath=False)
# def ceil(a):
#     """Element-wise ceil of a vector."""
#     return (math.ceil(a[0]), math.ceil(a[1]), math.ceil(a[2]))


# @cuda.jit(device=True, inline=True, fastmath=False)
# def min(a, b):
#     """shorter vector between a and b from the origin."""
#     return length(a) if length(a) < length(b) else length(b)


# @cuda.jit(device=True, inline=True, fastmath=False)
# def max(a, b):
#     """longer vector between a and b from the origin."""
#     return length(a) if length(a) > length(b) else length(b)


@device_jit
def linear_to_srgb(c):
    """Convert linear color component (float) to sRGB."""
    if c <= float32(0.0031308):
        return float32(12.92) * c
    return float32(1.055) * math.pow(c, ONE / float32(2.4)) - float32(0.055)


@device_jit
def srgb_to_linear(c):
    """Convert sRGB color component (float) to linear."""
    if c <= float32(0.04045):
        return c / float32(12.92)
    return math.pow((c + float32(0.055)) / float32(1.055), float32(2.4))
