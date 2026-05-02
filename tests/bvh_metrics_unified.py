#!/usr/bin/env python3
"""Unified BVH metrics test script.

Runs BVH metrics tests for all configurations across all scenes and
outputs results to .tests/ folder with a summary table.

Configurations:
  - sah-binning:  USE_SAH=True,  USE_BINNING=True
  - no-sah:       USE_SAH=False, USE_BINNING=True
  - no-binning:   USE_SAH=True,  USE_BINNING=False
  - naive:        USE_SAH=False, USE_BINNING=False

Scenes: bunny, box-spheres, dragon

Usage:
    ./tests/bvh_metrics_unified.py
    /home/bubakulus/miniforge3/bin/conda run -n raytracer python tests/bvh_metrics_unified.py
"""

import os
import sys
import re
import time
from datetime import datetime

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Directories
TESTS_DIR = os.path.join(project_root, ".tests")
SETTINGS_PATH = os.path.join(project_root, "src", "settings.py")

# Scenes and their resolutions
# Dragon uses lower resolution because no-binning BVH build is O(n^2) and very slow
SCENES = {
    "bunny": 1024,
    "box-spheres": 1024,
}

# BVH configurations: (name, use_sah, use_binning)
CONFIGS = [
    ("sah-binning", True, True),
    ("no-sah", False, True),
    ("no-binning", True, False),
    ("naive", False, False),
]

# Shared render parameters
SAMPLES = 16
MAX_BOUNCES = 16
RESOLUTION = 1024


def write_settings(scene, resolution, use_sah, use_binning):
    """Write settings.py for a given configuration."""
    with open(SETTINGS_PATH, "w") as f:
        f.write(f'DEVICE = "gpu"\n')
        f.write(f"CPU_DIMENSION = 500\n")
        f.write(f"GPU_DIMENSION = {resolution}\n")
        f.write(f'SCENE_NAME = "{scene}"\n')
        f.write(f"SAMPLES = {SAMPLES}\n")
        f.write(f"DENOISE = False\n")
        f.write(f"MAX_BOUNCES = {MAX_BOUNCES}\n")
        f.write(f'IMG_FORMAT = "jpg"\n')
        f.write(f"USE_BVH_CACHE = False\n")
        f.write(f"PRINT_STATS = False\n")
        f.write(f"RENDER_NON_BVH_STATS = False\n")
        f.write(f'TONEMAPPER = "none"\n')
        f.write(f"COLLECT_BVH_STATS = True\n")
        f.write(f"USE_SAH = {use_sah}\n")
        f.write(f"USE_BINNING = {use_binning}\n")


def run_raytracer():
    """Run the raytracer and capture stdout/stderr."""
    import subprocess

    cmd = [
        sys.executable,
        "-c",
        "import sys; sys.argv.append('--collect-bvh-stats'); from src import main; main()",
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = project_root + ":" + env.get("PYTHONPATH", "")

    result = subprocess.run(
        cmd,
        cwd=project_root,
        capture_output=True,
        text=True,
        env=env,
    )

    return result.stdout + result.stderr


def parse_metrics(output):
    """Parse BVH metrics from raytracer output."""
    metrics = {}

    # BVH construction time
    m = re.search(r"\[timing\]\s+bvh build\s+:\s+([\d.]+)\s+s", output)
    if m:
        metrics["construction_time"] = float(m.group(1))

    # Hit/miss pixels
    m = re.search(r"Hit pixels:\s+(\d[\d,]*)\s+\(([\d.]+)%\)", output)
    if m:
        metrics["hit_pixels"] = int(m.group(1).replace(",", ""))
        metrics["hit_pct"] = float(m.group(2))

    m = re.search(r"Miss pixels.*?:\s+(\d[\d,]*)\s+\(([\d.]+)%\)", output)
    if m:
        metrics["miss_pixels"] = int(m.group(1).replace(",", ""))
        metrics["miss_pct"] = float(m.group(2))

    # Overall stats: "  node_tests                    1.0    256.0       3.45"
    # followed by lines like "Overall mean node_tests:  3.45"
    overall_mean = re.search(r"Overall mean node_tests:\s+([\d.]+)", output)
    if overall_mean:
        metrics["mean_node_tests"] = float(overall_mean.group(1))

    overall_tri = re.search(r"Overall mean tri_tests:\s+([\d.]+)", output)
    if overall_tri:
        metrics["mean_tri_tests"] = float(overall_tri.group(1))

    overall_shadow = re.search(r"Overall mean shadow_tests:\s*([\d.]+)", output)
    if overall_shadow:
        metrics["mean_shadow_tests"] = float(overall_shadow.group(1))

    # Min/max/mean from the stats table:
    # "  node_tests                    1.0    256.0       3.45"
    # Use the same pattern for all three metrics
    for metric_name in ["node_tests", "tri_tests", "shadow_tests"]:
        pattern = rf"{metric_name}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
        m = re.search(pattern, output)
        if m:
            metrics[f"min_{metric_name}"] = float(m.group(1))
            metrics[f"max_{metric_name}"] = float(m.group(2))
            metrics[f"mean_{metric_name}"] = float(m.group(3))

    # Hit-pixel means
    hit_node = re.search(r"Hit-pixel mean node_tests:\s*([\d.]+)", output)
    if hit_node:
        metrics["hit_mean_node_tests"] = float(hit_node.group(1))

    hit_tri = re.search(r"Hit-pixel mean tri_tests:\s*([\d.]+)", output)
    if hit_tri:
        metrics["hit_mean_tri_tests"] = float(hit_tri.group(1))

    hit_shadow = re.search(r"Hit-pixel mean shadow_tests:\s*([\d.]+)", output)
    if hit_shadow:
        metrics["hit_mean_shadow_tests"] = float(hit_shadow.group(1))

    # Total tests
    total_node = re.search(r"Total node_tests:\s+([\d,]+)", output)
    if total_node:
        metrics["total_node_tests"] = int(total_node.group(1).replace(",", ""))

    total_tri = re.search(r"Total tri_tests:\s+([\d,]+)", output)
    if total_tri:
        metrics["total_tri_tests"] = int(total_tri.group(1).replace(",", ""))

    total_shadow = re.search(r"Total shadow_tests:\s+([\d,]+)", output)
    if total_shadow:
        metrics["total_shadow_tests"] = int(total_shadow.group(1).replace(",", ""))

    return metrics


def format_test_file(scene, config_name, metrics):
    """Format metrics into a test result string."""
    def format_count(value):
        if isinstance(value, int):
            return f"{value:,}"
        return "N/A"

    lines = []
    lines.append("=" * 70)
    lines.append(f"Scene: {scene}")
    lines.append(f"BVH Configuration: {config_name}")
    lines.append(f"Timestamp: {datetime.now().isoformat()}")
    lines.append("=" * 70)
    lines.append("")

    if "construction_time" in metrics:
        lines.append(f"BVH construction time (s): {metrics['construction_time']:.4f}")
    else:
        lines.append("BVH construction time (s): N/A")

    hit_pixels = metrics.get("hit_pixels", 0)
    miss_pixels = metrics.get("miss_pixels", 0)
    lines.append(f"Hit pixels: {hit_pixels:,} ({metrics.get('hit_pct', 0):.1f}%)")
    lines.append(f"Miss pixels: {miss_pixels:,} ({metrics.get('miss_pct', 0):.1f}%)")
    lines.append("")

    lines.append("GPU Metrics (all pixels):")
    lines.append("-" * 70)

    if "min_node_tests" in metrics:
        lines.append(
            f"  node_tests:    min={metrics['min_node_tests']:6.1f}  "
            f"max={metrics['max_node_tests']:6.1f}  "
            f"mean={metrics['mean_node_tests']:6.2f}"
        )
    else:
        lines.append("  node_tests:    N/A")

    if "min_tri_tests" in metrics:
        lines.append(
            f"  tri_tests:     min={metrics['min_tri_tests']:6.1f}  "
            f"max={metrics['max_tri_tests']:6.1f}  "
            f"mean={metrics['mean_tri_tests']:6.2f}"
        )
    else:
        lines.append("  tri_tests:     N/A")

    if "min_shadow_tests" in metrics:
        lines.append(
            f"  shadow_tests:  min={metrics['min_shadow_tests']:6.1f}  "
            f"max={metrics['max_shadow_tests']:6.1f}  "
            f"mean={metrics['mean_shadow_tests']:6.2f}"
        )
    else:
        lines.append("  shadow_tests:  N/A")

    lines.append("")
    lines.append("GPU Metrics (hit pixels only):")
    lines.append("-" * 70)

    if "hit_mean_node_tests" in metrics:
        lines.append(f"  node_tests:    mean={metrics['hit_mean_node_tests']:6.2f}")
        lines.append(f"  tri_tests:     mean={metrics['hit_mean_tri_tests']:6.2f}")
        lines.append(f"  shadow_tests:  mean={metrics['hit_mean_shadow_tests']:6.2f}")
    else:
        lines.append("  No hit pixels or metrics unavailable")

    lines.append("")
    lines.append(f"Total node_tests: {format_count(metrics.get('total_node_tests'))}")
    lines.append(f"Total tri_tests: {format_count(metrics.get('total_tri_tests'))}")
    lines.append(f"Total shadow_tests: {format_count(metrics.get('total_shadow_tests'))}")
    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines) + "\n"


def format_summary_table(results):
    """Format a markdown summary table from all results."""
    lines = []
    lines.append("# BVH Metrics Summary")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append("")
    lines.append(
        "| Scene | Config | Const (s) | Hit % | node_tests (mean) | tri_tests (mean) | shadow_tests (mean) |"
    )
    lines.append(
        "|-------|--------|-----------|-------|-------------------|------------------|---------------------|"
    )

    for scene, config_name, metrics in results:
        const = (
            f"{metrics.get('construction_time', 0):.4f}"
            if "construction_time" in metrics
            else "N/A"
        )
        hit_pct = f"{metrics.get('hit_pct', 0):.1f}"
        mean_node = f"{metrics.get('mean_node_tests', 0):.2f}"
        mean_tri = f"{metrics.get('mean_tri_tests', 0):.2f}"
        mean_shadow = f"{metrics.get('mean_shadow_tests', 0):.2f}"
        lines.append(
            f"| {scene} | {config_name} | {const} | {hit_pct}% | {mean_node} | {mean_tri} | {mean_shadow} |"
        )

    lines.append("")

    # Add detailed per-metric breakdown table
    lines.append("## Detailed Per-Metric Breakdown")
    lines.append("")

    for scene, config_name, metrics in results:
        lines.append(f"### {scene} - {config_name}")
        lines.append("")
        lines.append("| Metric | Min | Max | Mean |")
        lines.append("|--------|-----|-----|------|")

        if "min_node_tests" in metrics:
            lines.append(
                f"| node_tests | {metrics['min_node_tests']:.1f} | "
                f"{metrics['max_node_tests']:.1f} | {metrics['mean_node_tests']:.2f} |"
            )
        if "min_tri_tests" in metrics:
            lines.append(
                f"| tri_tests | {metrics['min_tri_tests']:.1f} | "
                f"{metrics['max_tri_tests']:.1f} | {metrics['mean_tri_tests']:.2f} |"
            )
        if "min_shadow_tests" in metrics:
            lines.append(
                f"| shadow_tests | {metrics['min_shadow_tests']:.1f} | "
                f"{metrics['max_shadow_tests']:.1f} | {metrics['mean_shadow_tests']:.2f} |"
            )

        lines.append("")

    return "\n".join(lines) + "\n"


def main():
    """Run all BVH metrics tests."""
    os.makedirs(TESTS_DIR, exist_ok=True)

    all_results = []
    start_total = time.time()

    print("=" * 70)
    print("  BVH METRICS UNIFIED TEST")
    print("=" * 70)
    print(f"Scenes: {', '.join(SCENES.keys())}")
    print(f"Configurations: {', '.join(c[0] for c in CONFIGS)}")
    print(f"Output directory: {TESTS_DIR}")
    print()

    for scene_name, resolution in SCENES.items():
        print(f"\n{'=' * 70}")
        print(f"  Scene: {scene_name} (resolution: {resolution}x{resolution})")
        print(f"{'=' * 70}")

        for config_name, use_sah, use_binning in CONFIGS:
            print(
                f"\n  Running: {config_name} (SAH={use_sah}, Binning={use_binning}) ...",
                end=" ",
                flush=True,
            )

            # Write settings
            write_settings(scene_name, resolution, use_sah, use_binning)

            # Run raytracer
            run_start = time.time()
            output = run_raytracer()
            run_time = time.time() - run_start

            # Parse metrics
            metrics = parse_metrics(output)
            metrics["scene"] = scene_name
            metrics["config"] = config_name

            # Format and write to file
            result_text = format_test_file(scene_name, config_name, metrics)
            output_file = os.path.join(TESTS_DIR, f"{scene_name}_{config_name}.txt")
            with open(output_file, "w") as f:
                f.write(result_text)

            # Print brief status
            const = metrics.get("construction_time", 0)
            mean_node = metrics.get("mean_node_tests", 0)
            mean_tri = metrics.get("mean_tri_tests", 0)
            print(
                f"done ({run_time:.1f}s, const={const:.4f}s, mean_node={mean_node:.1f}, mean_tri={mean_tri:.1f})"
            )

            all_results.append((scene_name, config_name, metrics))

    # Write summary
    summary_text = format_summary_table(all_results)
    summary_file = os.path.join(TESTS_DIR, "summary.txt")
    with open(summary_file, "w") as f:
        f.write(summary_text)

    total_time = time.time() - start_total

    print(f"\n{'=' * 70}")
    print(f"  All tests completed in {total_time:.1f}s")
    print(f"  Results saved to: {TESTS_DIR}/")
    print(f"  Summary: {summary_file}")
    print(f"{'=' * 70}")

    # Also print the summary to stdout for quick reference
    print(f"\n{summary_text}")


if __name__ == "__main__":
    main()
