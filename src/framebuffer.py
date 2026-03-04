from numpy import float32
from utils import device_jit
from utils.vec_utils import vec3, linear_to_srgb
from constants import ONE, HALF, UINT8_MAX_F, UINT8_MAX_I


@device_jit
def render_normals(n, fb, x, y):
    """[Debug]: output normals as colors"""
    r_val = (n[0] + ONE) * HALF
    g_val = (n[1] + ONE) * HALF
    b_val = (n[2] + ONE) * HALF

    fb[y, x, 0] = min(UINT8_MAX_I, int(r_val * UINT8_MAX_F))
    fb[y, x, 1] = min(UINT8_MAX_I, int(g_val * UINT8_MAX_F))
    fb[y, x, 2] = min(UINT8_MAX_I, int(b_val * UINT8_MAX_F))


@device_jit
def get_miss_color():
    """return dark gray background color"""
    val = float32(20.0) / UINT8_MAX_F
    return vec3(val, val, val)


@device_jit
def write_color_to_fb(cr, cg, cb, fb, x, y):
    # apply srgb transfer function and write to framebuffer
    r_srgb = int(linear_to_srgb(cr) * UINT8_MAX_F)
    g_srgb = int(linear_to_srgb(cg) * UINT8_MAX_F)
    b_srgb = int(linear_to_srgb(cb) * UINT8_MAX_F)

    fb[y, x, 0] = min(UINT8_MAX_I, r_srgb)
    fb[y, x, 1] = min(UINT8_MAX_I, g_srgb)
    fb[y, x, 2] = min(UINT8_MAX_I, b_srgb)
