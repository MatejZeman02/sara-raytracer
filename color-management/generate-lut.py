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

    # create a LUT_SIZE**3 grid of values from 0.0 to 1.0
    x = np.linspace(0, 1, LUT_SIZE)
    r, g, b = np.meshgrid(x, x, x, indexing="ij")
    grid_01 = np.stack((r, g, b), axis=-1)

    # Un-Log the grid to Linear ACEScg HDR values
    stops = grid_01 * (MAX_EV - MIN_EV) + MIN_EV
    linear_acescg = 2.0**stops

    # ACEScg -> Linear sRGB
    linear_srgb = colour.RGB_to_RGB(
        linear_acescg,
        colour.RGB_COLOURSPACES["ACEScg"],
        colour.RGB_COLOURSPACES["sRGB"],
        chromatic_adaptation_transform="CAT02",  # white point adaptation
    )

    # (OKLAB) GAMUT COMPRESSION
    # Convert Linear sRGB to Oklab
    XYZ = colour.RGB_to_XYZ(
        linear_srgb,
        colour.RGB_COLOURSPACES["sRGB"].whitepoint,
        colour.RGB_COLOURSPACES["sRGB"].whitepoint,
        colour.RGB_COLOURSPACES["sRGB"].matrix_RGB_to_XYZ,
    )
    oklab = colour.XYZ_to_Oklab(XYZ)

    # Vectorized Binary Search for Chroma Compression
    min_chan = np.min(linear_srgb, axis=-1, keepdims=True)
    needs_compression = min_chan < 0

    low = np.zeros_like(min_chan)
    high = np.ones_like(min_chan)

    # 15 iterations yields plenty of precision for float32
    for _ in range(15):
        mid = (low + high) / 2.0

        # Scale the 'a' and 'b' channels (Chroma)
        oklab_test = oklab.copy()
        oklab_test[..., 1:3] *= np.where(needs_compression, mid, 1.0)

        # Convert back to test if it fits
        XYZ_test = colour.Oklab_to_XYZ(oklab_test)
        rgb_test = colour.XYZ_to_RGB(
            XYZ_test,
            colour.RGB_COLOURSPACES["sRGB"].whitepoint,
            colour.RGB_COLOURSPACES["sRGB"].whitepoint,
            colour.RGB_COLOURSPACES["sRGB"].matrix_XYZ_to_RGB,
        )

        min_test = np.min(rgb_test, axis=-1, keepdims=True)

        # If still negative, we have too much Chroma (move high bound down)
        high = np.where(min_test < 0, mid, high)
        # If positive, we can afford more Chroma (move low bound up)
        low = np.where(min_test >= 0, mid, low)

    # Apply the final solved Chroma scale
    oklab_final = oklab.copy()
    oklab_final[..., 1:3] *= np.where(needs_compression, low, 1.0)

    XYZ_final = colour.Oklab_to_XYZ(oklab_final)
    gamut_mapped = colour.XYZ_to_RGB(
        XYZ_final,
        colour.RGB_COLOURSPACES["sRGB"].whitepoint,
        colour.RGB_COLOURSPACES["sRGB"].whitepoint,
        colour.RGB_COLOURSPACES["sRGB"].matrix_XYZ_to_RGB,
    )

    # TONE MAPPING: Narkowicz filmic curve on linear sRGB.
    # 0.6 pre-scale aligns the simplified curve with the Academy RRT baseline.
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14

    exposed_gamut = gamut_mapped * 0.6
    mapped_rgb = (exposed_gamut * (a * exposed_gamut + b)) / (
        exposed_gamut * (c * exposed_gamut + d) + e
    )
    mapped_rgb = np.clip(mapped_rgb, 0.0, 1.0)

    # 2.2 decode to linearize — output is linear SDR [0, 1].
    # The renderer's gamma lut will then apply the sRGB gamma curve to produce
    # the final uint8 image.
    tonemapped_linear = mapped_rgb**2.2

    # save as .npy for Numba
    np.save("color-management/acescg_to_srgb.npy", tonemapped_linear.astype(np.float32))
    print("Saved as 'color-management/acescg_to_srgb.npy'")


if __name__ == "__main__":
    build_custom_aces_lut()
