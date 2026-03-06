from numpy import float32
from utils import device_jit
from utils.vec_utils import vec3, linear_to_srgb
from constants import ONE, HALF, UINT8_MAX_F, UINT8_MAX_I, ZERO


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
def aces_narkowicz_tonemap(x):
    """aces filmic tonemapping curve approximate"""
    assert x >= ZERO

    a = float32(2.51)
    b = float32(0.03)
    c = float32(2.43)
    d = float32(0.59)
    e = float32(0.14)

    mapped = (x * (a * x + b)) / (x * (c * x + d) + e)
    mapped = max(ZERO, min(ONE, mapped))

    return mapped


@device_jit
def write_color_to_fb(cr, cg, cb, fb, x, y):
    assert x >= 0
    assert y >= 0

    # tone map the linear hdr color to sdr
    cr_mapped = aces_narkowicz_tonemap(cr)
    cg_mapped = aces_narkowicz_tonemap(cg)
    cb_mapped = aces_narkowicz_tonemap(cb)

    # apply srgb transfer function and write to framebuffer
    r_srgb = int(linear_to_srgb(cr_mapped) * UINT8_MAX_F)
    g_srgb = int(linear_to_srgb(cg_mapped) * UINT8_MAX_F)
    b_srgb = int(linear_to_srgb(cb_mapped) * UINT8_MAX_F)

    # clipping limit is no longer needed due to tonemap clamping
    fb[y, x, 0] = r_srgb
    fb[y, x, 1] = g_srgb
    fb[y, x, 2] = b_srgb
