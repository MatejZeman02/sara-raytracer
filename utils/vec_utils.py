"""Math vector utilities for device-agnostic kernels."""

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


@device_jit
def neg(a):
    """Negate a 3D vector."""
    return (float32(-a[0]), float32(-a[1]), float32(-a[2]))


@device_jit
def apply_3d_lut_gpu(r, g, b, lut):
    """Trilinear interpolation of a 32x32x32 LUT on the GPU.

    Expects linear-log exposure-adjusted input — camera exposure is applied
    by the caller before invoking this function.
    """
    MIN_EV = float32(-10.0)
    MAX_EV = float32(10.0)

    # Log2 encode directly
    r_log = (math.log2(max(r, float32(1e-8))) - MIN_EV) / (MAX_EV - MIN_EV)
    g_log = (math.log2(max(g, float32(1e-8))) - MIN_EV) / (MAX_EV - MIN_EV)
    b_log = (math.log2(max(b, float32(1e-8))) - MIN_EV) / (MAX_EV - MIN_EV)

    r_log = min(ONE, max(ZERO, r_log))
    g_log = min(ONE, max(ZERO, g_log))
    b_log = min(ONE, max(ZERO, b_log))

    # Scale to grid
    size = float32(lut.shape[0] - 1)
    x = r_log * size
    y = g_log * size
    z = b_log * size

    x0 = int(x)
    x1 = min(x0 + 1, lut.shape[0] - 1)
    y0 = int(y)
    y1 = min(y0 + 1, lut.shape[0] - 1)
    z0 = int(z)
    z1 = min(z0 + 1, lut.shape[0] - 1)

    xd = x - x0
    yd = y - y0
    zd = z - z0

    # Trilinear blend (X axis)
    c00_r = lut[x0, y0, z0, 0] * (ONE - xd) + lut[x1, y0, z0, 0] * xd
    c00_g = lut[x0, y0, z0, 1] * (ONE - xd) + lut[x1, y0, z0, 1] * xd
    c00_b = lut[x0, y0, z0, 2] * (ONE - xd) + lut[x1, y0, z0, 2] * xd

    c10_r = lut[x0, y1, z0, 0] * (ONE - xd) + lut[x1, y1, z0, 0] * xd
    c10_g = lut[x0, y1, z0, 1] * (ONE - xd) + lut[x1, y1, z0, 1] * xd
    c10_b = lut[x0, y1, z0, 2] * (ONE - xd) + lut[x1, y1, z0, 2] * xd

    c01_r = lut[x0, y0, z1, 0] * (ONE - xd) + lut[x1, y0, z1, 0] * xd
    c01_g = lut[x0, y0, z1, 1] * (ONE - xd) + lut[x1, y0, z1, 1] * xd
    c01_b = lut[x0, y0, z1, 2] * (ONE - xd) + lut[x1, y0, z1, 2] * xd

    c11_r = lut[x0, y1, z1, 0] * (ONE - xd) + lut[x1, y1, z1, 0] * xd
    c11_g = lut[x0, y1, z1, 1] * (ONE - xd) + lut[x1, y1, z1, 1] * xd
    c11_b = lut[x0, y1, z1, 2] * (ONE - xd) + lut[x1, y1, z1, 2] * xd

    # Y axis
    c0_r = c00_r * (ONE - yd) + c10_r * yd
    c0_g = c00_g * (ONE - yd) + c10_g * yd
    c0_b = c00_b * (ONE - yd) + c10_b * yd

    c1_r = c01_r * (ONE - yd) + c11_r * yd
    c1_g = c01_g * (ONE - yd) + c11_g * yd
    c1_b = c01_b * (ONE - yd) + c11_b * yd

    # Z axis
    out_r = c0_r * (ONE - zd) + c1_r * zd
    out_g = c0_g * (ONE - zd) + c1_g * zd
    out_b = c0_b * (ONE - zd) + c1_b * zd

    return out_r, out_g, out_b
