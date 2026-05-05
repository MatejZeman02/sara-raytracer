"""Render each tonemapper and save to src/output/hw06/.

Usage:
    # Render bunny scene (default, acescg materials):
    conda run -n raytracer python tests/render_tonemappers.py

    # Render box-spheres scene (rec709 materials):
    SCENE_NAME=box-spheres conda run -n raytracer python tests/render_tonemappers.py

    Each tonemapper runs with CLI args:
      --tonemapper <name>  --scene <scene>  --samples 16  --denoise true  --format jpg
"""

import os
import shutil
import subprocess
import tempfile

PROJECT = "/home/bubakulus/work/mata/prog/pg1"
OUT_DIR = os.path.join(PROJECT, "src", "output", "hw06")
SRC_OUT = os.path.join(PROJECT, "src", "output")
CONDA = "/home/bubakulus/miniforge3/bin/conda run -n raytracer"

MAPPERS = [
    ("custom-aces", "custom-aces.jpg"),
    ("narkowicz", "narkowitcz.jpg"),
    ("khronos", "khronos.jpg"),
    ("hill", "hill.jpg"),
    ("none", "none.jpg"),
    ("magenta", "magenta.jpg"),
]

SCENE_NAME = os.environ.get("SCENE_NAME", "bunny")


def render_one(tonemapper, out_name):
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
    cmd = f'{CONDA} python "{script.name}" --tonemapper "{tonemapper}" --scene "{SCENE_NAME}" --samples 16 --denoise true --format jpg'
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, cwd=PROJECT
    )
    os.remove(script.name)
    return result.stdout + result.stderr


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Rendering {len(MAPPERS)} tonemappers -> {OUT_DIR}\n")
    timings = {}
    for tonemapper, out_name in MAPPERS:
        print(f"[{tonemapper}] -> {out_name} ...", flush=True)
        output = render_one(tonemapper, out_name)
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
        if f.endswith(".jpg"):
            print(f"  {f}")


if __name__ == "__main__":
    main()
