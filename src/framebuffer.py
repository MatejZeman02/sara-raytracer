import math
import os
import numpy as np
from numpy import float32, uint8
from numba import njit, prange, cuda
from utils import device_jit
from utils.vec_utils import apply_3d_lut_gpu
from .constants import ONE, HALF, UINT8_MAX_F, UINT8_MAX_I, ZERO
from .settings import settings

# Extract value at import time so Numba sees a concrete string
_TONEMAPPER = settings.TONEMAPPER


def create_gamma_lut():
    """Pre-calculates precise sRGB uint8 values for 65536 steps."""
    cache_path = f"color-management/srgb_gamma.npz"

    # load existing if already cached
    if os.path.exists(cache_path):
        lut = np.load(cache_path, allow_pickle=True)
        return lut["lut"]

    LUT_SIZE = 65536  # 16-bit
    lut = np.zeros(LUT_SIZE, dtype=np.uint8)

    for i in prange(LUT_SIZE):
        val = i / float(LUT_SIZE - 1)

        # Exact sRGB curve
        if val <= 0.0031308:
            srgb = 12.92 * val
        else:
            srgb = 1.055 * (val ** (1.0 / 2.4)) - 0.055

        lut[i] = int(max(0.0, min(1.0, srgb)) * 255.0)

    # cache for future runs
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, lut=lut)
    return lut


@device_jit
def render_normals(n, fb, x, y):
    """[Debug]: output normals as colors"""
    r_val = (n[0] + ONE) * HALF
    g_val = (n[1] + ONE) * HALF
    b_val = (n[2] + ONE) * HALF

    fb[y, x, 0] = min(UINT8_MAX_I, int(r_val * UINT8_MAX_F))
    fb[y, x, 1] = min(UINT8_MAX_I, int(g_val * UINT8_MAX_F))
    fb[y, x, 2] = min(UINT8_MAX_I, int(b_val * UINT8_MAX_F))


@njit(fastmath=True)
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


if settings.DEVICE == "gpu":
    khronos_tonemap = njit(fastmath=True)
else:
    khronos_tonemap = device_jit


@khronos_tonemap
def khronos_pbr_neutral_tonemapper(r, g, b):
    """Leaves values below 0.8 mostly unchanged"""
    # constants defined by the khronos spec
    start_compression = float32(0.8) - float32(0.04)
    desaturation = float32(0.15)

    # if in SDR, do nothing
    peak = max(r, max(g, b))
    if peak < start_compression:
        return r, g, b

    # apply a small offset to black levels
    min_channel = min(r, min(g, b))
    if min_channel < float32(0.08):
        offset = min_channel - (float32(6.25) * min_channel * min_channel)
    else:
        offset = float32(0.04)

    r -= offset
    g -= offset
    b -= offset

    # compress highlights
    d = ONE - start_compression
    new_peak = ONE - d * d / (peak + d - start_compression)

    # scale by compressed peak
    scale = new_peak / peak
    r *= scale
    g *= scale
    b *= scale

    # desaturation factor for extremes
    desat_factor = ONE - ONE / (desaturation * (peak - new_peak) + ONE)

    # mix the color towards white (new_peak) based on desat_factor
    r = r * (ONE - desat_factor) + new_peak * desat_factor
    g = g * (ONE - desat_factor) + new_peak * desat_factor
    b = b * (ONE - desat_factor) + new_peak * desat_factor

    return r, g, b


@device_jit
def write_hdr_to_fb(cr, cg, cb, fb_hdr, x, y):
    """write raw float32 accumulated color into HDR framebuffer."""
    assert x >= 0
    assert y >= 0
    fb_hdr[y, x, 0] = cr
    fb_hdr[y, x, 1] = cg
    fb_hdr[y, x, 2] = cb


METHOD = "none"  # "none", "khronos" or "aces"


@njit(fastmath=True)
def linear_to_srgb(c):
    if c <= float32(0.0031308):
        return float32(12.92) * c
    return float32(1.055) * math.pow(c, ONE / float32(2.4)) - float32(0.055)


@njit(fastmath=True)
def magenta_debug_tonemap(r, g, b):
    """debug mode: visualize hdr values exceeding display gamut as magenta."""
    # check if any channel exceeds display gamut
    # FIXME: doesn't work with AP1 / ACEScg primaries as working space?
    peak = max(r, max(g, b))
    if peak > ONE:
        # use the maximum channel as the luminance proxy
        lum = max(r, max(g, b))
        # magenta = full red + full blue, no green
        return lum, ZERO, lum
    # values within display gamut pass through unchanged
    return r, g, b


@njit(parallel=True, fastmath=True)
def tonemap_hdr_to_sdr(fb_hdr, width, height):
    """apply tonemapping in-place to an hdr float32 buffer.

    the buffer is rewritten in-place with float32 sdr values in [0, 1].
    mode is determined by TONEMAPPER setting from settings.py.
    """
    assert width > 0
    assert height > 0
    assert fb_hdr.shape[0] == height
    assert fb_hdr.shape[1] == width

    for y in prange(height):
        for x in range(width):
            for c in range(3):
                # check for NaNs or negative values:
                assert (
                    fb_hdr[y, x, c] >= ZERO
                ), f"Negative color value in HDR buffer at x: {x}, y: {y}, channel: {c}"
                assert (
                    fb_hdr[y, x, c] == fb_hdr[y, x, c]
                ), f"NaN color value in HDR buffer at x: {x}, y: {y}, channel: {c}"

            cr = fb_hdr[y, x, 0]
            cg = fb_hdr[y, x, 1]
            cb = fb_hdr[y, x, 2]

            if _TONEMAPPER == "none":
                cr_mapped, cg_mapped, cb_mapped = cr, cg, cb
            elif _TONEMAPPER == "aces":
                cr_mapped = aces_narkowicz_tonemap(cr)
                cg_mapped = aces_narkowicz_tonemap(cg)
                cb_mapped = aces_narkowicz_tonemap(cb)
            elif _TONEMAPPER == "magenta":
                cr_mapped, cg_mapped, cb_mapped = magenta_debug_tonemap(cr, cg, cb)
            else:
                # default: khronos pbr neutral tonemapper
                cr_mapped, cg_mapped, cb_mapped = khronos_pbr_neutral_tonemapper(
                    cr, cg, cb
                )

            # keep OIDN in LDR-safe range before denoising
            if cr_mapped < ZERO:
                cr_mapped = ZERO
            elif cr_mapped > ONE:
                cr_mapped = ONE

            if cg_mapped < ZERO:
                cg_mapped = ZERO
            elif cg_mapped > ONE:
                cg_mapped = ONE

            if cb_mapped < ZERO:
                cb_mapped = ZERO
            elif cb_mapped > ONE:
                cb_mapped = ONE

            fb_hdr[y, x, 0] = cr_mapped
            fb_hdr[y, x, 1] = cg_mapped
            fb_hdr[y, x, 2] = cb_mapped


@cuda.jit
def tonemap_kernel(fb_hdr, fb_ldr, lut, width, height):
    x, y = cuda.grid(2)
    if x < width and y < height:
        r = fb_hdr[y, x, 0]
        g = fb_hdr[y, x, 1]
        b = fb_hdr[y, x, 2]

        # apply the LUT
        out_r, out_g, out_b = apply_3d_lut_gpu(r, g, b, lut)

        fb_ldr[y, x, 0] = out_r
        fb_ldr[y, x, 1] = out_g
        fb_ldr[y, x, 2] = out_b


@cuda.jit
def postprocess_full_gpu_kernel(fb_hdr, out_uint8, lut, gamma_lut, width, height):
    """apply 3D lut + gamma lut and output uint8 directly on gpu."""
    x, y = cuda.grid(2)
    if x < width and y < height:
        r = fb_hdr[y, x, 0]
        g = fb_hdr[y, x, 1]
        b = fb_hdr[y, x, 2]

        out_r, out_g, out_b = apply_3d_lut_gpu(r, g, b, lut)

        # clamp and map linear sdr to final uint8 via cached gamma lut
        if out_r < 0.0:
            out_r = 0.0
        elif out_r > 1.0:
            out_r = 1.0
        if out_g < 0.0:
            out_g = 0.0
        elif out_g > 1.0:
            out_g = 1.0
        if out_b < 0.0:
            out_b = 0.0
        elif out_b > 1.0:
            out_b = 1.0

        lut_max = gamma_lut.shape[0] - 1
        out_uint8[y, x, 0] = gamma_lut[int(out_r * lut_max)]
        out_uint8[y, x, 1] = gamma_lut[int(out_g * lut_max)]
        out_uint8[y, x, 2] = gamma_lut[int(out_b * lut_max)]


@njit(parallel=True, fastmath=True)
def postprocess_sdr_to_u8(fb_ldr, out_uint8, gamma_lut, width, height):
    """apply sRGB gamma to an SDR float32 buffer and write uint8 pixels.

    Runs on the CPU (host numpy arrays) after tonemapping and denoising.
    """
    for i in prange(height * width):
        y = i // width
        x = i - y * width
        for c in range(3):
            val = fb_ldr[y, x, c]
            val_clamped = max(0.0, min(1.0, val))
            index = int(val_clamped * (len(gamma_lut) - 1))
            out_uint8[y, x, c] = gamma_lut[index]
