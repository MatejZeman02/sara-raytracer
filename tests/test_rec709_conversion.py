"""
Test Rec.709 -> ACEScg conversion for box-advanced scene.

Key finding: ACEScg is a RELATIVE working space where D65 and D60 white
are both represented as [1,1,1]. The _ACESCG_MAT_T matrix is the correct
sRGB (Rec.709 D65) -> ACEScg (AP1) conversion.

Tests:
1. brick_diffuse.png (white) -> sRGB->linear -> _ACESCG_MAT_T = white in ACEScg
2. Material Kd=[0.8,0.8,0.8] -> _ACESCG_MAT_T = [0.8,0.8,0.8] (neutral)
3. Test render with white back_wall (no texture, Kd=white) -> wall is white
"""

import os
import sys
import shutil
import subprocess
import numpy as np
from PIL import Image

# Absolute paths
SCENE_DIR = "/home/bubakulus/work/mata/prog/pg1/scenes/box-advanced"
MTL_PATH = os.path.join(SCENE_DIR, "box-advanced.mtl")
SETTINGS_PATH = "/home/bubakulus/work/mata/prog/pg1/src/settings.py"
OUTPUT_DIR = "/home/bubakulus/work/mata/prog/pg1/src/output"
PROJECT_DIR = "/home/bubakulus/work/mata/prog/pg1"
SRC_DIR = os.path.join(PROJECT_DIR, "src")

# The CORRECT matrix (sRGB/Rec.709 D65 -> ACEScg AP1)
# White [1,1,1] -> [1.0, 1.0, 1.0] in ACEScg
_ACESCG_MAT_T = np.array(
    [
        [0.613097, 0.070194, 0.020615],
        [0.339523, 0.916355, 0.109569],
        [0.047379, 0.013451, 0.869816],
    ],
    dtype=np.float32,
)

# WRONG matrix (chained XYZ + Bradford)
_REC709_TO_XYZ_MAT = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float32,
)
_WRONG_REC709_MAT_T = (
    np.tensordot(_ACESCG_MAT_T, _REC709_TO_XYZ_MAT.T, axes=([1], [0]))
    .reshape(3, 3)
    .astype(np.float32)
)


def srgb_to_linear(rgb):
    rgb = np.maximum(rgb, 0.0)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def apply_mat(tex_arr, mat_t):
    """Apply 3x3 matrix to texture (row-vector convention)."""
    return np.tensordot(tex_arr, mat_t.T, axes=([2], [1])).reshape(tex_arr.shape)


def apply_mat_vec(rgb, mat_t):
    """Apply 3x3 matrix to RGB vector (row-vector: rgb @ mat)."""
    return np.dot(rgb, mat_t)


def test_texture_conversion():
    """Test that brick_diffuse.png (white) stays white after conversion."""
    print("=" * 60)
    print("TEXTURE CONVERSION TEST")
    print("=" * 60)

    tex_path = os.path.join(SCENE_DIR, "brick_diffuse.png")
    if not os.path.exists(tex_path):
        print(f"  WARNING: {tex_path} not found")
        return

    img = Image.open(tex_path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    srgb_mean = arr.mean()
    print(f"  Original sRGB mean:  {srgb_mean:.6f}")

    # sRGB -> linear
    lin = srgb_to_linear(arr)
    lin_mean = lin.mean()
    print(f"  After sRGB->linear:  {lin_mean:.6f}")

    # Convert with CORRECT _ACESCG_MAT_T (sRGB_to_ACEScg)
    converted_correct = apply_mat(lin, _ACESCG_MAT_T)
    cur_mean = converted_correct.mean()
    print(f"  After _ACESCG_MAT_T (correct): {cur_mean:.6f} (should be ~1.0)")
    print(f"  -> Deviation: {abs(cur_mean - 1.0):.6f}")
    print(
        f"  Per-channel: R={converted_correct[:,:,0].mean():.6f}, "
        f"G={converted_correct[:,:,1].mean():.6f}, "
        f"B={converted_correct[:,:,2].mean():.6f}"
    )

    # Convert with WRONG chained matrix
    converted_wrong = apply_mat(lin, _WRONG_REC709_MAT_T)
    wrng_mean = converted_wrong.mean()
    print(f"  After _WRONG_REC709_MAT_T:     {wrng_mean:.6f} (should be ~1.0)")
    print(f"  -> Deviation: {abs(wrng_mean - 1.0):.6f}")
    print(
        f"  Per-channel: R={converted_wrong[:,:,0].mean():.6f}, "
        f"G={converted_wrong[:,:,1].mean():.6f}, "
        f"B={converted_wrong[:,:,2].mean():.6f}"
    )

    # Per-pixel max difference
    diff = np.abs(converted_correct - converted_wrong).max()
    print(f"\n  Max per-pixel difference between correct and wrong: {diff:.6f}")


def test_material_conversion():
    """Test material color conversion (back_wall Kd=[0.8,0.8,0.8])."""
    print("\n" + "=" * 60)
    print("MATERIAL CONVERSION TEST")
    print("=" * 60)

    kdb = np.array([[0.8, 0.8, 0.8]], dtype=np.float32)

    # CORRECT: apply _ACESCG_MAT_T directly
    converted_correct = apply_mat_vec(kdb, _ACESCG_MAT_T)
    print(f"  Kd=[0.8, 0.8, 0.8] rec709 ->")
    print(f"    Correct (_ACESCG_MAT_T):      {converted_correct[0]}")
    print(
        f"    Deviation from [0.8,0.8,0.8]: "
        f"R={abs(converted_correct[0,0]-0.8):.6f}, "
        f"G={abs(converted_correct[0,1]-0.8):.6f}, "
        f"B={abs(converted_correct[0,2]-0.8):.6f}"
    )

    # WRONG: no conversion (what my code does)
    print(f"    No conversion (my bug):       {kdb[0]} (wrong - no matrix)")

    # WRONG: chained matrix
    converted_wrong = apply_mat_vec(kdb, _WRONG_REC709_MAT_T)
    print(f"    Wrong (_REC709_TO_ACESCG):    {converted_wrong[0]}")

    # left_wall Kd = [1.0, 0.0, 0.0] (pure red)
    kdr = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    print(f"\n  Kd=[1.0, 0.0, 0.0] (pure rec709 red) ->")
    print(f"    Correct:  {apply_mat_vec(kdr, _ACESCG_MAT_T)[0]}")
    print(f"    Wrong:    {apply_mat_vec(kdr, _WRONG_REC709_MAT_T)[0]}")


def test_render_white_material():
    """Render with a modified scene where back_wall is pure white (no texture)."""
    print("\n" + "=" * 60)
    print("TEST RENDER: White back wall (no texture, Kd=white)")
    print("=" * 60)

    # Backup MTL
    mtl_backup_path = MTL_PATH + ".backup"
    shutil.copy2(MTL_PATH, mtl_backup_path)

    try:
        # Modify back_wall to white with no texture
        with open(MTL_PATH, "r") as f:
            mtl_content = f.read()

        mtl_content = mtl_content.replace(
            "newmtl back_wall\nNs 0.0\nKd 0.8 0.8 0.8\nKs 0.0 0.0 0.0\nKe 0.0 0.0 0.0\nNi 1.0\nd 1.0\nillum 1\nmap_Kd brick_diffuse.png",
            "newmtl back_wall\nNs 0.0\nKd 1.0 1.0 1.0\nKs 0.0 0.0 0.0\nKe 0.0 0.0 0.0\nNi 1.0\nd 1.0\nillum 1",
        )

        with open(MTL_PATH, "w") as f:
            f.write(mtl_content)

        print("  Modified back_wall: Kd=[1,1,1], no texture")

        # Backup and modify settings
        settings_backup = SETTINGS_PATH + ".backup"
        shutil.copy2(SETTINGS_PATH, settings_backup)

        with open(SETTINGS_PATH, "r") as f:
            settings_content = f.read()

        settings_content = settings_content.replace(
            'SCENE_NAME = "bunny"', 'SCENE_NAME = "box-advanced"'
        )
        settings_content = settings_content.replace("SAMPLES = 1000", "SAMPLES = 500")
        settings_content = settings_content.replace('DEVICE = "gpu"', 'DEVICE = "cpu"')
        settings_content = settings_content.replace("DENOISE = True", "DENOISE = False")
        settings_content = settings_content.replace("TONEMAP = True", "TONEMAP = False")

        with open(SETTINGS_PATH, "w") as f:
            f.write(settings_content)

        print("  Settings: box-advanced, CPU, 500 samples, no denoise, no tonemap")

        # Run render via subprocess
        cmd = f"import sys; sys.path.insert(0, '{SRC_DIR}'); import main"
        result = subprocess.run(
            [sys.executable, "-c", cmd], cwd=PROJECT_DIR, capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"  Render failed:\n{result.stderr}")
            return
        print("  Render complete.")

        # Check output
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_files = sorted(
            [f for f in os.listdir(OUTPUT_DIR) if f.startswith("output")],
            key=lambda f: os.path.getmtime(os.path.join(OUTPUT_DIR, f)),
            reverse=True,
        )
        if output_files:
            out_path = os.path.join(OUTPUT_DIR, output_files[0])
            img = Image.open(out_path).convert("RGB")
            arr = np.asarray(img, dtype=np.float32) / 255.0

            h, w = arr.shape[:2]
            wall_region = arr[
                int(h * 0.15) : int(h * 0.45), int(w * 0.05) : int(w * 0.95)
            ]
            wall_mean = wall_region.mean()

            print(f"\n  Output: {out_path}")
            print(
                f"  Wall mean color: R={wall_mean[0]:.6f}, G={wall_mean[1]:.6f}, B={wall_mean[2]:.6f}"
            )
            print(
                f"  Wall std:        R={wall_region[:,:,0].std():.6f}, "
                f"G={wall_region[:,:,1].std():.6f}, "
                f"B={wall_region[:,:,2].std():.6f}"
            )
            print(
                f"  Overall mean:    R={arr[:,:,0].mean():.6f}, "
                f"G={arr[:,:,1].mean():.6f}, "
                f"B={arr[:,:,2].mean():.6f}"
            )

            r_diff = abs(wall_mean[0] - wall_mean[1])
            g_diff = abs(wall_mean[1] - wall_mean[2])
            print(f"\n  Wall neutrality (R-G={r_diff:.6f}, G-B={g_diff:.6f})")
            if r_diff < 0.05 and g_diff < 0.05:
                print("  -> PASS: wall is neutral (white/gray)")
            else:
                print("  -> FAIL: wall has color cast")

    finally:
        shutil.copy2(mtl_backup_path, MTL_PATH)
        os.remove(mtl_backup_path)
        shutil.copy2(settings_backup, SETTINGS_PATH)
        os.remove(settings_backup)
        print("\n  Restored MTL and settings files.")


if __name__ == "__main__":
    test_texture_conversion()
    test_material_conversion()
    test_render_white_material()
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
