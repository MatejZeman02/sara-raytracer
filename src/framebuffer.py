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


@njit(fastmath=True)
def narkowicz_tonemap(x):
    """Krzysztof Narkowicz ACES fit tonemapper, returning linear SDR.
    https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/

    0.6 pre-scale aligns the simplified curve with the Academy RRT baseline.
    ** 2.2 decode linearizes the output so the gamma lut can apply the final
    sRGB gamma curve without double-encoding.
    negative values (from out-of-gamut CSC) are clamped to zero.
    """

    # clamp negative values (from out-of-gamut ACEScg CSC) to zero
    if x < ZERO:
        x = ZERO

    a = float32(2.51)
    b = float32(0.03)
    c = float32(2.43)
    d = float32(0.59)
    e = float32(0.14)

    # 0.6 pre-scale to match the Academy ACES reference curve
    x = x * float32(0.6)

    mapped = (x * (a * x + b)) / (x * (c * x + d) + e)

    mapped = max(ZERO, min(ONE, mapped))

    # linearize output (decode baked gamma) for gamma lut
    return math.pow(mapped, float32(2.2))


@njit(fastmath=True)
def hill_tonemap(x):
    """Stephen Hill ACES approximation tonemapper, returning linear SDR.

    0.6 pre-scale aligns the simplified curve with the Academy RRT baseline.
    ** 2.2 decode linearizes the output so the gamma lut can apply the final
    sRGB gamma curve without double-encoding.
    negative values (from out-of-gamut CSC) are clamped to zero.
    """
    # clamp negative values (from out-of-gamut ACEScg CSC) to zero
    if x < ZERO:
        x = ZERO

    # 0.6 pre-scale to match the Academy ACES reference curve
    x = x * float32(0.6)

    # coefficients for hill curve
    a = float32(0.0245786)
    b = float32(0.000090537)
    c = float32(0.983729)
    d = float32(0.4329510)
    e = float32(0.238081)

    # quadratic numerator over quadratic denominator
    mapped = (x * (x + a) - b) / (x * (c * x + d) + e)
    mapped = max(ZERO, min(ONE, mapped))

    # linearize output (decode baked gamma) for gamma lut
    return mapped**2.2


if settings.DEVICE == "gpu":
    khronos_tonemap = njit(fastmath=True)
else:
    khronos_tonemap = device_jit


@khronos_tonemap
def khronos_pbr_neutral_tonemapper(r, g, b):
    """Leaves values below 0.8 mostly unchanged"""
    # avoid negative values from out-of-gamut CSC by clamping to zero
    r = max(float32(0.0), r)
    g = max(float32(0.0), g)
    b = max(float32(0.0), b)
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


@njit(fastmath=True)
def magenta_debug_tonemap(r, g, b):
    """debug mode: visualize hdr values exceeding display gamut as magenta."""
    # check if any channel exceeds display gamut
    peak = max(r, max(g, b))
    if peak > ONE:
        # use the maximum channel as the luminance proxy
        lum = max(r, max(g, b))
        # magenta = full red + full blue, no green
        return lum, ZERO, lum
    # values within display gamut pass through unchanged
    return r, g, b


@njit(parallel=True, fastmath=True)
def acescg_to_linear_srgb(fb_hdr, width, height):
    """Convert ACEScg (ap1) to linear sRGB using industry-standard CSC matrix.

    Handles both chromatic adaptation (D60 → D65) and primary conversion.
    Modifies fb_hdr in-place.
    """
    # ACEScg → linear sRGB (Rec.709) chromatic adaptation matrix
    for y in prange(height):
        for x in range(width):
            r = fb_hdr[y, x, 0]
            g = fb_hdr[y, x, 1]
            b = fb_hdr[y, x, 2]
            fb_hdr[y, x, 0] = (
                r * float32(1.7050796)
                + g * float32(-0.6218677)
                + b * float32(-0.0832119)
            )
            fb_hdr[y, x, 1] = (
                r * float32(-0.1302553)
                + g * float32(1.1408020)
                + b * float32(-0.0105467)
            )
            fb_hdr[y, x, 2] = (
                r * float32(-0.0240075)
                + g * float32(-0.1289677)
                + b * float32(1.1529752)
            )


@njit(parallel=True, fastmath=True)
def tonemap_hdr_to_sdr(fb_hdr, width, height, exposure_mul):
    """apply the unified output transform pipeline in-place to an hdr float32 buffer.

    single-pass: exposure → csc (acescg → linear srgb) → tonemapper → clamp.
    all materials are in acescg working space. the buffer is rewritten in-place
    with float32 sdr values in [0, 1]. camera exposure is a linear multiplier
    applied before gamut conversion.
    """
    assert width > 0
    assert height > 0
    assert fb_hdr.shape[0] == height
    assert fb_hdr.shape[1] == width

    for y in prange(height):
        for x in range(width):
            for c in range(3):
                assert (
                    fb_hdr[y, x, c] >= ZERO
                ), f"negative color value in hdr buffer at x: {x}, y: {y}, channel: {c}"
                assert (
                    fb_hdr[y, x, c] == fb_hdr[y, x, c]
                ), f"nan color value in hdr buffer at x: {x}, y: {y}, channel: {c}"

            # read once — pixel stays in l1 cache for the full pipeline
            cr = fb_hdr[y, x, 0]
            cg = fb_hdr[y, x, 1]
            cb = fb_hdr[y, x, 2]

            # exposure
            cr *= exposure_mul
            cg *= exposure_mul
            cb *= exposure_mul

            # csc: acescg (ap1 d60) → linear srgb (rec.709 d65)
            sr = (
                cr * float32(1.7050796)
                + cg * float32(-0.6218677)
                + cb * float32(-0.0832119)
            )
            sg = (
                cr * float32(-0.1302553)
                + cg * float32(1.1408020)
                + cb * float32(-0.0105467)
            )
            sb = (
                cr * float32(-0.0240075)
                + cg * float32(-0.1289677)
                + cb * float32(1.1529752)
            )

            # tonemap + clamp (ldr-safe range for oidn denoiser)
            tr, tg, tb = _apply_tonemap(sr, sg, sb)
            tr = max(ZERO, min(ONE, tr))
            tg = max(ZERO, min(ONE, tg))
            tb = max(ZERO, min(ONE, tb))

            fb_hdr[y, x, 0] = tr
            fb_hdr[y, x, 1] = tg
            fb_hdr[y, x, 2] = tb


@njit(fastmath=True)
def _apply_tonemap(cr, cg, cb):
    """apply the selected film stock tonemapper to linear HDR values."""
    if _TONEMAPPER == "none":
        return cr, cg, cb
    elif _TONEMAPPER == "narkowicz":
        return narkowicz_tonemap(cr), narkowicz_tonemap(cg), narkowicz_tonemap(cb)
    elif _TONEMAPPER == "magenta":
        return magenta_debug_tonemap(cr, cg, cb)
    elif _TONEMAPPER == "hill":
        return hill_tonemap(cr), hill_tonemap(cg), hill_tonemap(cb)
    else:
        # default: khronos pbr neutral tonemapper
        return khronos_pbr_neutral_tonemapper(cr, cg, cb)


@cuda.jit
def tonemap_kernel(fb_hdr, fb_ldr, lut, width, height, exposure_mul):
    x, y = cuda.grid(2)
    if x < width and y < height:
        # apply camera exposure
        r = fb_hdr[y, x, 0] * exposure_mul
        g = fb_hdr[y, x, 1] * exposure_mul
        b = fb_hdr[y, x, 2] * exposure_mul

        # 3d lut: csc (acescg → linear srgb) + tonemap (0.6 pre-scale + curve + 2.2 decode) + gamma encode
        out_r, out_g, out_b = apply_3d_lut_gpu(r, g, b, lut)

        fb_ldr[y, x, 0] = out_r
        fb_ldr[y, x, 1] = out_g
        fb_ldr[y, x, 2] = out_b


@cuda.jit
def postprocess_full_gpu_kernel(
    fb_hdr, out_uint8, lut, gamma_lut, width, height, exposure_mul
):
    """apply camera exposure, 3d lut + gamma lut and output uint8 on gpu.

    the 3d lut performs: csc (acescg → linear srgb) → tonemap (0.6 pre-scale + curve + 2.2 decode) → gamma encode.
    output from 3d lut is linear sdr.
    the gamma lut maps the final sdr values to uint8.
    """
    x, y = cuda.grid(2)
    if x < width and y < height:
        # apply camera exposure
        r = fb_hdr[y, x, 0] * exposure_mul
        g = fb_hdr[y, x, 1] * exposure_mul
        b = fb_hdr[y, x, 2] * exposure_mul

        # 3d lut: csc + tonemap (with 0.6 pre-scale + 2.2 decode) + gamma encode
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
