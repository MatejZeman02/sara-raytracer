from numpy import float32, uint8
from numba import njit, prange
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
def aces_narkowicz_tonemap(x):
    """aces filmic tonemapping curve approximate"""
    assert x >= ZERO

    a = float32(2.51)  # aces_coeff_a
    b = float32(0.03)  # aces_coeff_b
    c = float32(2.43)  # aces_coeff_c
    d = float32(0.59)  # aces_coeff_d
    e = float32(0.14)  # aces_coeff_e

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


@device_jit
def write_hdr_to_fb(cr, cg, cb, fb_hdr, x, y):
    """write raw float32 accumulated color into HDR framebuffer."""
    assert x >= 0
    assert y >= 0
    fb_hdr[y, x, 0] = cr
    fb_hdr[y, x, 1] = cg
    fb_hdr[y, x, 2] = cb


@njit(parallel=False, fastmath=False)
def postprocess_hdr(fb_hdr, out, width, height):
    """apply ACES filmic tonemap + sRGB gamma to denoised HDR buffer, write uint8.

    Runs on the CPU (host numpy arrays) after OIDN denoising so it is always
    compiled as a plain Numba njit kernel regardless of the render DEVICE.
    """
    assert width > 0
    assert height > 0
    assert fb_hdr.shape[0] == height
    assert fb_hdr.shape[1] == width
    assert out.shape[0] == height
    assert out.shape[1] == width

    for y in prange(height):
        for x in range(width):
            for c in range(3):
                # check for NaNs or negative values
                assert fb_hdr[y, x, c] >= ZERO, f"Negative color value in HDR buffer at x: {x}, y: {y}, channel: {c}"
                assert fb_hdr[y, x, c] == fb_hdr[y, x, c], f"NaN color value in HDR buffer at x: {x}, y: {y}, channel: {c}"
                v = fb_hdr[y, x, c]  # channel_value
                # ACES Narkowicz approximation (matches aces_narkowicz_tonemap)
                v = (v * (float32(2.51) * v + float32(0.03))) / (
                    v * (float32(2.43) * v + float32(0.59)) + float32(0.14)
                )
                if v < float32(0.0):
                    v = float32(0.0)
                elif v > float32(1.0):
                    v = float32(1.0)
                # linear to sRGB (IEC 61966-2-1)
                if v <= float32(0.0031308):
                    v = float32(12.92) * v
                else:
                    v = float32(1.055) * (v ** float32(1.0 / 2.4)) - float32(0.055)
                out[y, x, c] = uint8(int(v * float32(255.0)))
