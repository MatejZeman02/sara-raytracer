"""Render each tonemapper and save to src/output/hw06/.

Usage:
    # Render bunny scene (default, acescg materials):
    conda run -n raytracer python tests/render_tonemappers.py

    # Render box-spheres scene (rec709 materials):
    SCENE_NAME=box-spheres conda run -n raytracer python tests/render_tonemappers.py

    Each tonemapper runs with CLI args:
      --tonemapper <name>  --scene <scene>  --samples 16  --denoise true  --format jpg
"""

import argparse
import os
import shutil
import subprocess
import tempfile

PROJECT = "/home/bubakulus/work/mata/prog/pg1"
OUT_DIR = os.path.join(PROJECT, "src", "output", "hw06")
SRC_OUT = os.path.join(PROJECT, "src", "output")
CONDA = "/home/bubakulus/miniforge3/bin/conda run -n raytracer"

MAPPERS = [
    ("custom", "custom.jpg"),
    ("narkowicz", "narkowicz.jpg"),
    ("khronos", "khronos.jpg"),
    ("hill", "hill.jpg"),
    ("none", "none.jpg"),
    ("magenta", "magenta.jpg"),
]


def parse_args():
    p = argparse.ArgumentParser(description="Render all tonemappers and save results.")
    p.add_argument(
        "--scene",
        default=os.environ.get("SCENE_NAME", "bunny"),
        help="Scene directory name (default: $SCENE_NAME or 'bunny')",
    )
    p.add_argument(
        "--samples",
        type=int,
        default=30000,
        help="Samples per pixel",
    )
    p.add_argument(
        "--bounces",
        type=int,
        default=16,
        help="Max ray bounces (default: 16)",
    )
    p.add_argument(
        "--denoise",
        choices=["true", "false"],
        default="true",
        help="Enable denoiser (default: true)",
    )
    p.add_argument(
        "--format",
        choices=["jpg", "png"],
        default="jpg",
        help="Output format (default: jpg)",
    )
    return p.parse_args()


def render_one(tonemapper, out_name, args):
    """Run one tonemapper in a fresh python process via conda with CLI args."""
    script = tempfile.NamedTemporaryFile(suffix=".py", dir=OUT_DIR, delete=False)
    script.write(f"""
import os, sys, shutil
sys.path.insert(0, r"{PROJECT}")
from src import main
main()
src = os.path.join(r"{SRC_OUT}", "output.jpg")
dst = os.path.join(r"{OUT_DIR}", "{out_name}")
shutil.copy2(src, dst)
print(f"[saved] {out_name}")
""".encode())
    script.close()
    cmd = (
        f'{CONDA} python "{script.name}" '
        f'--tonemapper "{tonemapper}" '
        f'--scene "{args.scene}" '
        f"--samples {args.samples} "
        f"--bounces {args.bounces} "
        f"--denoise {args.denoise} "
        f"--format {args.format}"
    )
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, cwd=PROJECT
    )
    os.remove(script.name)
    return result.stdout + result.stderr


def main():
    args = parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Scene: {args.scene} | Samples: {args.samples} | Bounces: {args.bounces}")
    print(f"Rendering {len(MAPPERS)} tonemappers -> {OUT_DIR}\n")
    timings = {}
    for tonemapper, out_name in MAPPERS:
        print(f"[{tonemapper}] -> {out_name} ...", flush=True)
        output = render_one(tonemapper, out_name, args)
        for line in output.strip().splitlines():
            if "total" in line and "timing" in line:
                timings[out_name] = line.strip()
                print(f"  {line.strip()}")
        print()

    print("=" * 60)
    print("TIMING SUMMARY")
    print("=" * 60)
    for out_name, timing in timings.items():
        print(f"  {out_name:<16} {timing}")
    print()
    print("Files in hw06/:")
    for f in sorted(os.listdir(OUT_DIR)):
        if f.endswith(".jpg") or f.endswith(".png"):
            print(f"  {f}")


if __name__ == "__main__":
    main()
