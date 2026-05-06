"""This script generates a 3D LUT that maps from ACEScg to sRGB, using an Oklab gamut compression and ACES 2.0 style tonemapping."""

import numpy as np
import colour
import os

# settings:
LUT_SIZE = 32
# The HDR range the LUT will cover.
# -10 to +10 EV covers everything from pitch black to white sun.
MIN_EV = -10.0
MAX_EV = 10.0


def build_custom_aces_lut():
    print(f"Generating {LUT_SIZE}x{LUT_SIZE}x{LUT_SIZE} LUT...")

    # grid generation
    # build a regular 3D grid in [0, 1] and un-log it to full ACEScg HDR
    x = np.linspace(0, 1, LUT_SIZE)
    r, g, b = np.meshgrid(x, x, x, indexing="ij")
    grid_01 = np.stack((r, g, b), axis=-1)

    stops = grid_01 * (MAX_EV - MIN_EV) + MIN_EV
    linear_acescg = 2.0**stops

    # acescg to linear srgb
    # transform colour space with cat02 chromatic adaptation
    linear_srgb = colour.RGB_to_RGB(
        linear_acescg,
        colour.RGB_COLOURSPACES["ACEScg"],
        colour.RGB_COLOURSPACES["sRGB"],
        chromatic_adaptation_transform="CAT02",
    )

    # asymmetrical crosstalk
    # simulates physical film emulsion where colors bleed unevenly
    # red bleeds into green, green into blue, blue into red
    crosstalk_matrix = np.array(
        [[0.85, 0.10, 0.05], [0.05, 0.85, 0.10], [0.10, 0.05, 0.85]], dtype=np.float32
    )

    linear_srgb = np.dot(linear_srgb, crosstalk_matrix.T)

    # oklab gamut compression
    # convert to oklab to detect and squash out-of-gamut chroma
    XYZ = colour.RGB_to_XYZ(linear_srgb, "sRGB")
    oklab = colour.XYZ_to_Oklab(XYZ)

    # vectorized binary search for chroma compression
    min_chan = np.min(linear_srgb, axis=-1, keepdims=True)
    needs_compression = min_chan < 0

    low = np.zeros_like(min_chan)
    high = np.ones_like(min_chan)

    # 15 iterations yields plenty of precision for float32
    for _ in range(15):
        mid = (low + high) / 2.0
        oklab_test = oklab.copy()
        oklab_test[..., 1:3] *= np.where(needs_compression, mid, 1.0)

        XYZ_test = colour.Oklab_to_XYZ(oklab_test)
        rgb_test = colour.XYZ_to_RGB(XYZ_test, "sRGB")

        min_test = np.min(rgb_test, axis=-1, keepdims=True)
        high = np.where(min_test < 0, mid, high)
        low = np.where(min_test >= 0, mid, low)

    # apply the final solved chroma scale
    oklab_final = oklab.copy()
    oklab_final[..., 1:3] *= np.where(needs_compression, low, 1.0)

    # path to white highlight desaturation
    # gradually reduce chroma as lightness rises toward pure white
    l = oklab_final[..., 0]
    desat_start = 1.5
    desat_end = 2.0

    # linear falloff from 1.0 (at desat_start) to 0.0 (at desat_end)
    factor = 1.0 - np.clip((l - desat_start) / (desat_end - desat_start), 0.0, 1.0)
    # hermite smoothstep for a photographic, gradual transition
    factor = factor * factor * (3.0 - 2.0 * factor)
    oklab_final[..., 1:3] *= factor[..., np.newaxis]

    XYZ_final = colour.Oklab_to_XYZ(oklab_final)
    gamut_mapped = colour.XYZ_to_RGB(XYZ_final, "sRGB")

    # uchimura tonemapping
    # extract achromatic norm to preserve hue during tonemapping,
    # then apply the uchimura curve to luminance before scaling back
    norm = np.max(gamut_mapped, axis=-1, keepdims=True)

    # exposure pre-scale (equivalent to Academy RRT 0.6 factor)
    exposed_norm = norm * 0.6

    # ensure no negative values enter the math curve
    x = np.maximum(exposed_norm, 0.0)

    # uchimura curve parameters
    p = 1.0  # max display brightness
    a = 1.1  # contrast
    m = 0.22  # linear section start
    l_len = 0.4  # linear section length
    c = 1.2  # black tightness and toe drop
    b = 0.0  # black offset

    # uchimura math logic — precompute breakpoints
    l0 = ((p - m) * l_len) / a
    l0_cap = m - m / a
    s0 = m + l0
    s1 = m + a * l0
    c2 = (a * p) / (p - s1)
    cp = -c2 / p

    # piecewise blend weights
    w0 = 1.0 - np.clip((x - l0_cap) / (m - l0_cap), 0.0, 1.0)
    w2 = np.clip((x - s0) / (p - s0), 0.0, 1.0)
    w1 = 1.0 - w0 - w2

    # safe division for the toe math
    safe_m = np.where(m == 0.0, 1e-6, m)
    t = safe_m * (x / safe_m) ** c + b
    s = p - (p - s1) * np.exp(cp * (x - s0))
    l_curve = safe_m + a * (x - safe_m)

    mapped_norm = t * w0 + l_curve * w1 + s * w2

    # calculate scale ratio to apply tonemap back to rgb channels
    scale = np.zeros_like(norm)
    np.divide(mapped_norm, norm, out=scale, where=norm > 0)

    mapped_rgb = gamut_mapped * scale
    mapped_rgb = np.clip(mapped_rgb, 0.0, 1.0)

    # output log decode
    tonemapped_linear = mapped_rgb**2.2

    # save as .npy for Numba
    np.save("color-management/acescg_to_srgb.npy", tonemapped_linear.astype(np.float32))
    print("Saved as 'color-management/acescg_to_srgb.npy'")


if __name__ == "__main__":
    build_custom_aces_lut()
