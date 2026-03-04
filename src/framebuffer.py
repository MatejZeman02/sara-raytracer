import numpy as np
from utils import device_jit
from utils.vec_utils import vec3, linear_to_srgb
from constants import ONE, HALF


@device_jit
def render_normals(n, fb, x, y):
    """[Debug]: output normals as colors"""
    r_val = (n[0] + ONE) * HALF
    g_val = (n[1] + ONE) * HALF
    b_val = (n[2] + ONE) * HALF

    fb[y, x, 0] = min(255, int(r_val * np.float32(255.0)))
    fb[y, x, 1] = min(255, int(g_val * np.float32(255.0)))
    fb[y, x, 2] = min(255, int(b_val * np.float32(255.0)))


@device_jit
def get_miss_color():
    """return dark gray background color"""
    val = np.float32(20.0) / np.float32(255.0)
    return vec3(val, val, val)


@device_jit
def write_color_to_fb(cr, cg, cb, fb, x, y):
    # apply srgb transfer function and write to framebuffer
    r_srgb = int(linear_to_srgb(cr) * 255)
    g_srgb = int(linear_to_srgb(cg) * 255)
    b_srgb = int(linear_to_srgb(cb) * 255)

    fb[y, x, 0] = min(255, r_srgb)
    fb[y, x, 1] = min(255, g_srgb)
    fb[y, x, 2] = min(255, b_srgb)
