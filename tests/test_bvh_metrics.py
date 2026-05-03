#!/usr/bin/env python3
"""Unified BVH metrics test script.

Runs BVH metrics tests for all configurations across all scenes and
outputs results to .tests/ folder with a summary table.

Configurations:
  - sah-binning:  USE_SAH=True,  USE_BINNING=True
  - median-split:  USE_SAH=False, USE_BINNING=True
  - no-binning:    USE_SAH=True,  USE_BINNING=False

Scenes: bunny, box-spheres

Usage:
    ./tests/bvh_metrics_unified.py
    /home/bubakulus/miniforge3/bin/conda run -n raytracer python tests/bvh_metrics_unified.py

    # Update README with parsed metrics:
    /home/bubakulus/miniforge3/bin/conda run -n raytracer python tests/test_bvh_metrics.py --write-readme
"""

import argparse
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
    ("median-split", False, True),
    ("no-binning", True, False),
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

    # Resolution (for query time estimation)
    m = re.search(r"Resolution:\s+(\d+)\s+x\s+(\d+)\s+\(([\d,]+)\s+pixels\)", output)
    if m:
        metrics["num_pixels"] = int(m.group(3).replace(",", ""))

    # Render time (for query time estimation)
    m = re.search(r"Render time:\s+([\d.]+)\s+s", output)
    if m:
        metrics["render_time"] = float(m.group(1))

    # Hit/miss pixels
    m = re.search(r"Hit pixels:\s+(\d[\d,]*)\s+\(([\d.]+)%\)", output)
    if m:
        metrics["hit_pixels"] = int(m.group(1).replace(",", ""))
        metrics["hit_pct"] = float(m.group(2))

    m = re.search(r"Miss pixels.*?:\s+(\d[\d,]*)\s+\(([\d.]+)%\)", output)
    if m:
        metrics["miss_pixels"] = int(m.group(1).replace(",", ""))
        metrics["miss_pct"] = float(m.group(2))

    # Construction metrics from BVH structure
    for key, pattern in [
        ("total_nodes", r"Total nodes:\s+([\d,]+)"),
        ("internal_nodes", r"Internal nodes:\s+([\d,]+)"),
        ("leaf_nodes", r"Leaf nodes:\s+([\d,]+)"),
    ]:
        m = re.search(pattern, output)
        if m:
            metrics[key] = int(m.group(1).replace(",", ""))

    m = re.search(
        r"Leaf depth \(min/max/mean\):\s+(\d+)\s*/\s+(\d+)\s*/\s+([\d.]+)", output
    )
    if m:
        metrics["leaf_depth_min"] = int(m.group(1))
        metrics["leaf_depth_max"] = int(m.group(2))
        metrics["leaf_depth_mean"] = float(m.group(3))

    m = re.search(r"Prims/leaf \(min/max\):\s+(\d+)\s*/\s+(\d+)", output)
    if m:
        metrics["leaf_prims_min"] = int(m.group(1))
        metrics["leaf_prims_max"] = int(m.group(2))

    m = re.search(r"Prims/leaf \(mean\):\s+([\d.]+)", output)
    if m:
        metrics["leaf_prims_mean"] = float(m.group(1))

    m = re.search(r"Memory:\s+([\d,]+)\s+bytes\s*\(([\d.]+)\s+KB\)", output)
    if m:
        metrics["memory_bytes"] = int(m.group(1).replace(",", ""))
        metrics["memory_kb"] = float(m.group(2))

    # Traversal metrics: min/max/mean from the stats table
    for metric_name in [
        "node_tests",
        "tri_tests",
        "shadow_tests",
        "traverse_tests",
        "query_depth",
    ]:
        pattern = rf"{metric_name}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
        m = re.search(pattern, output)
        if m:
            metrics[f"min_{metric_name}"] = float(m.group(1))
            metrics[f"max_{metric_name}"] = float(m.group(2))
            metrics[f"mean_{metric_name}"] = float(m.group(3))

    # Overall mean from summary lines
    overall_mean = re.search(r"Overall mean node_tests:\s+([\d.]+)", output)
    if overall_mean:
        metrics["mean_node_tests"] = float(overall_mean.group(1))

    overall_tri = re.search(r"Overall mean tri_tests:\s+([\d.]+)", output)
    if overall_tri:
        metrics["mean_tri_tests"] = float(overall_tri.group(1))

    overall_shadow = re.search(r"Overall mean shadow_tests:\s*([\d.]+)", output)
    if overall_shadow:
        metrics["mean_shadow_tests"] = float(overall_shadow.group(1))

    overall_traverse = re.search(r"Overall mean traverse_tests:\s*([\d.]+)", output)
    if overall_traverse:
        metrics["mean_traverse_tests"] = float(overall_traverse.group(1))

    overall_qd = re.search(r"Overall mean query_depth:\s*([\d.]+)", output)
    if overall_qd:
        metrics["mean_query_depth"] = float(overall_qd.group(1))

    # Hit-pixel means
    for metric_name in ["node_tests", "tri_tests", "shadow_tests"]:
        hit_m = re.search(rf"Hit-pixel mean {metric_name}:\s*([\d.]+)", output)
        if hit_m:
            metrics[f"hit_mean_{metric_name}"] = float(hit_m.group(1))

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

    # Query time estimation (mean only)
    # query_time = (render_time - construction_time) / total_queries
    # where total_queries = num_pixels * SAMPLES
    # Note: render_time must be > construction_time for this to be meaningful
    if (
        "render_time" in metrics
        and "construction_time" in metrics
        and "num_pixels" in metrics
        and metrics["render_time"] > metrics["construction_time"]
    ):
        render_time = metrics["render_time"]
        construction_time = metrics["construction_time"]
        num_pixels = metrics["num_pixels"]
        total_queries = num_pixels * SAMPLES
        query_time = (render_time - construction_time) / total_queries
        metrics["query_time_mean"] = query_time

    return metrics


def format_count(value):
    """Format an integer count with thousand separators."""
    if value is None:
        return "N/A"
    return f"{value:,}"


def format_test_file(scene, config_name, metrics):
    """Format metrics into a test result string."""

    def fmt(value, default="N/A"):
        if isinstance(value, int):
            return f"{value:,}"
        if isinstance(value, float):
            return f"{value:.2f}"
        return default

    lines = []
    lines.append("=" * 70)
    lines.append(f"Scene: {scene}")
    lines.append(f"BVH Configuration: {config_name}")
    lines.append(f"Timestamp: {datetime.now().isoformat()}")
    lines.append("=" * 70)
    lines.append("")

    # Construction metrics
    lines.append("CONSTRUCTION METRICS:")
    lines.append("-" * 70)
    if "construction_time" in metrics:
        lines.append(f"  Construction time (s):    {metrics['construction_time']:.4f}")
    else:
        lines.append("  Construction time (s):    N/A")

    if "total_nodes" in metrics:
        lines.append(f"  Total nodes:              {metrics['total_nodes']:,}")
    else:
        lines.append("  Total nodes:              N/A")

    if "internal_nodes" in metrics:
        lines.append(f"  Internal nodes:           {metrics['internal_nodes']:,}")
        lines.append(f"  Leaf nodes:               {metrics['leaf_nodes']:,}")
    else:
        lines.append("  Internal nodes:           N/A")
        lines.append("  Leaf nodes:               N/A")

    if "leaf_depth_min" in metrics:
        lines.append(
            f"  Leaf depth (min/max):     {metrics['leaf_depth_min']} / {metrics['leaf_depth_max']}"
        )
        lines.append(
            f"  Prims/leaf (min/max):     {metrics['leaf_prims_min']} / {metrics['leaf_prims_max']}"
        )
        lines.append(f"  Prims/leaf (mean):        {metrics['leaf_prims_mean']:.2f}")
        lines.append(
            f"  Memory:                   {fmt(metrics.get('memory_bytes', 0), 'N/A')} bytes"
        )
    else:
        lines.append("  Leaf depth (min/max):     N/A")
        lines.append("  Prims/leaf (min/max):     N/A")
        lines.append("  Prims/leaf (mean):        N/A")
        lines.append("  Memory:                   N/A")

    lines.append("")

    hit_pixels = metrics.get("hit_pixels", 0)
    miss_pixels = metrics.get("miss_pixels", 0)
    lines.append(f"Hit pixels: {hit_pixels:,} ({metrics.get('hit_pct', 0):.1f}%)")
    lines.append(f"Miss pixels: {miss_pixels:,} ({metrics.get('miss_pct', 0):.1f}%)")
    lines.append("")

    # Traversal metrics (all pixels)
    lines.append("TRAVERSAL METRICS (all pixels):")
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

    if "min_traverse_tests" in metrics:
        lines.append(
            f"  traverse_tests: min={metrics['min_traverse_tests']:5.1f}  "
            f"max={metrics['max_traverse_tests']:6.1f}  "
            f"mean={metrics['mean_traverse_tests']:6.2f}"
        )
    else:
        lines.append("  traverse_tests: N/A")

    if "min_query_depth" in metrics:
        lines.append(
            f"  query_depth:   min={metrics['min_query_depth']:6.1f}  "
            f"max={metrics['max_query_depth']:6.1f}  "
            f"mean={metrics['mean_query_depth']:6.2f}"
        )
    else:
        lines.append("  query_depth:   N/A")

    lines.append("")
    lines.append("TRAVERSAL METRICS (hit pixels only):")
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
    lines.append(
        f"Total shadow_tests: {format_count(metrics.get('total_shadow_tests'))}"
    )
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

    # Construction summary table
    lines.append("## Construction Metrics")
    lines.append("")
    lines.append(
        "| Scene | Config | Const (s) | Nodes | Internal | Leaves | Mem (KB) | Leaf depth (min/max/mean) | Prims/leaf (min/max/mean) |"
    )
    lines.append(
        "|-------|--------|-----------|-------|----------|--------|----------|-------------------------|---------------------------|"
    )

    for scene, config_name, metrics in results:
        const = (
            f"{metrics.get('construction_time', 0):.4f}"
            if "construction_time" in metrics
            else "N/A"
        )
        total_nodes = metrics.get("total_nodes", 0)
        internal = metrics.get("internal_nodes", 0)
        leaves = metrics.get("leaf_nodes", 0)
        mem_kb = (
            f"{metrics.get('memory_kb', 0):.1f}" if "memory_kb" in metrics else "N/A"
        )
        ld = (
            f"{metrics.get('leaf_depth_min', 0)}/{metrics.get('leaf_depth_max', 0)}/{metrics.get('leaf_depth_mean', 0):.1f}"
            if "leaf_depth_min" in metrics
            else "N/A"
        )
        lp = (
            f"{metrics.get('leaf_prims_min', 0)}/{metrics.get('leaf_prims_max', 0)}/{metrics.get('leaf_prims_mean', 0):.1f}"
            if "leaf_prims_min" in metrics
            else "N/A"
        )
        lines.append(
            f"| {scene} | {config_name} | {const} | {total_nodes:,} | {internal:,} | {leaves:,} | {mem_kb} | {ld} | {lp} |"
        )

    lines.append("")

    # Traversal summary table
    lines.append("## Traversal Metrics (mean per pixel)")
    lines.append("")
    lines.append(
        "| Scene | Config | Hit % | node_tests | tri_tests | shadow_tests | traverse_tests | query_depth | query_time |"
    )
    lines.append(
        "|-------|--------|-------|------------|-----------|--------------|----------------|-------------|------------|"
    )

    for scene, config_name, metrics in results:
        hit_pct = f"{metrics.get('hit_pct', 0):.1f}"
        mean_node = f"{metrics.get('mean_node_tests', 0):.2f}"
        mean_tri = f"{metrics.get('mean_tri_tests', 0):.2f}"
        mean_shadow = f"{metrics.get('mean_shadow_tests', 0):.2f}"
        mean_traverse = f"{metrics.get('mean_traverse_tests', 0):.2f}"
        mean_qd = f"{metrics.get('mean_query_depth', 0):.2f}"
        qt = (
            f"{metrics.get('query_time_mean', 0):.6f}"
            if "query_time_mean" in metrics
            else "N/A"
        )
        lines.append(
            f"| {scene} | {config_name} | {hit_pct}% | {mean_node} | {mean_tri} | {mean_shadow} | {mean_traverse} | {mean_qd} | {qt} |"
        )

    lines.append("")

    # Add detailed per-metric breakdown table
    lines.append("## Detailed Per-Metric Breakdown")
    lines.append("")

    for scene, config_name, metrics in results:
        lines.append(f"### {scene} - {config_name}")
        lines.append("")

        # Construction sub-table
        lines.append("### Construction")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Total nodes | {_fmt(metrics, 'total_nodes')} |")
        lines.append(f"| Internal nodes | {_fmt(metrics, 'internal_nodes')} |")
        lines.append(f"| Leaf nodes | {_fmt(metrics, 'leaf_nodes')} |")
        ld_min = _fmt(metrics, "leaf_depth_min")
        ld_max = _fmt(metrics, "leaf_depth_max")
        ld = f"{ld_min} / {ld_max}" if "leaf_depth_min" in metrics else "N/A"
        lines.append(f"| Leaf depth (min/max) | {ld} |")
        lp_min = _fmt(metrics, "leaf_prims_min")
        lp_max = _fmt(metrics, "leaf_prims_max")
        lp = f"{lp_min} / {lp_max}" if "leaf_prims_min" in metrics else "N/A"
        lines.append(f"| Prims/leaf (min/max) | {lp} |")
        lines.append(
            f"| Prims/leaf (mean) | {_fmt(metrics, 'leaf_prims_mean', fmt_str='.2f')} |"
        )
        if "memory_kb" in metrics:
            mem_bytes = _fmt(metrics, "memory_bytes", fmt_str=",")
            mem_kb = _fmt(metrics, "memory_kb", fmt_str=".1f")
            lines.append(f"| Memory | {mem_bytes} bytes ({mem_kb} KB) |")
        else:
            lines.append("| Memory | N/A |")
        lines.append("")

        # Traversal sub-table
        lines.append("### Traversal (all pixels)")
        lines.append("")
        lines.append("| Metric | Min | Max | Mean |")
        lines.append("|--------|-----|-----|------|")

        for metric_name in [
            "node_tests",
            "tri_tests",
            "shadow_tests",
            "traverse_tests",
            "query_depth",
        ]:
            if f"min_{metric_name}" in metrics:
                lines.append(
                    f"| {metric_name} | {metrics[f'min_{metric_name}']:.1f} | "
                    f"{metrics[f'max_{metric_name}']:.1f} | {metrics[f'mean_{metric_name}']:.2f} |"
                )

        lines.append("")

    return "\n".join(lines) + "\n"


def _fmt(m, key, default="—", fmt_str=None):
    """Format a metric value with fallback for missing keys."""
    if key not in m:
        return default
    val = m[key]
    if fmt_str is not None:
        return f"{val:{fmt_str}}"
    return str(val)


def _constr_row(scene, config, m):
    """One row for the Construction Metrics summary table."""
    const = _fmt(m, "construction_time", fmt_str=".4f")
    nodes = _fmt(m, "total_nodes", fmt_str=",")
    internal = _fmt(m, "internal_nodes", fmt_str=",")
    leaves = _fmt(m, "leaf_nodes", fmt_str=",")
    ld_min = _fmt(m, "leaf_depth_min")
    ld_max = _fmt(m, "leaf_depth_max")
    ld_mean = _fmt(m, "leaf_depth_mean", fmt_str=".1f")
    ld = f"{ld_min} / {ld_max} / {ld_mean}" if "leaf_depth_min" in m else "—"
    lp_min = _fmt(m, "leaf_prims_min")
    lp_max = _fmt(m, "leaf_prims_max")
    lp_mean = _fmt(m, "leaf_prims_mean", fmt_str=".1f")
    lp = f"{lp_min} / {lp_max} / {lp_mean}" if "leaf_prims_min" in m else "—"
    return (
        f"| {scene:<13} | {config:<10} | {const:>9} | {nodes:>7} | {internal:>10} | "
        f"{leaves:>8} | {ld:<25} | {lp:<25} |"
    )


def _trav_row(scene, config, m):
    """One row for the Traversal Metrics summary table."""
    hit_pct = _fmt(m, "hit_pct", fmt_str=".1f")
    mean_node = _fmt(m, "mean_node_tests", fmt_str=".2f")
    mean_tri = _fmt(m, "mean_tri_tests", fmt_str=".2f")
    mean_shadow = _fmt(m, "mean_shadow_tests", fmt_str=".2f")
    mean_traverse = _fmt(m, "mean_traverse_tests", fmt_str=".2f")
    mean_qd = _fmt(m, "mean_query_depth", fmt_str=".2f")
    qt = _fmt(m, "query_time_mean", fmt_str=".6f")
    return (
        f"| {scene:<13} | {config:<10} | {hit_pct:>5} | {mean_node:>10} | "
        f"{mean_tri:>9} | {mean_shadow:>12} | {mean_traverse:>14} | {mean_qd:>9} | {qt:>10} |"
    )


def _summary_row(scene, config, m):
    """One row for the main BVH Performance summary table."""
    const = _fmt(m, "construction_time", fmt_str=".4f")
    mean_node = _fmt(m, "mean_node_tests", fmt_str=".2f")
    mean_tri = _fmt(m, "mean_tri_tests", fmt_str=".2f")
    mean_shadow = _fmt(m, "mean_shadow_tests", fmt_str=".2f")
    return (
        f"| {scene:<13} | {config:<13} | {const:>14} | {mean_node:>17} | "
        f"{mean_tri:>14} | {mean_shadow:>17} |"
    )


def _build_constr_table(m):
    """Build the Construction sub-table for the Detailed Breakdown."""
    nodes = _fmt(m, "total_nodes")
    internal = _fmt(m, "internal_nodes")
    leaves = _fmt(m, "leaf_nodes")
    ld = (
        f"{_fmt(m, 'leaf_depth_min')} / {_fmt(m, 'leaf_depth_max')}"
        if "leaf_depth_min" in m
        else "—"
    )
    lp = (
        f"{_fmt(m, 'leaf_prims_min')} / {_fmt(m, 'leaf_prims_max')}"
        if "leaf_prims_min" in m
        else "—"
    )
    lp_mean = _fmt(m, "leaf_prims_mean", fmt_str=".2f")
    mem_bytes = _fmt(m, "memory_bytes", fmt_str=",")
    mem_kb = _fmt(m, "memory_kb", fmt_str=".1f")
    return (
        f"| Total nodes          | {nodes:<30} |\n"
        f"| Internal nodes       | {internal:<30} |\n"
        f"| Leaf nodes           | {leaves:<30} |\n"
        f"| Leaf depth (min/max) | {ld:<30} |\n"
        f"| Prims/leaf (min/max) | {lp:<30} |\n"
        f"| Prims/leaf (mean)    | {lp_mean:<30} |\n"
        f"| Memory               | {mem_bytes} bytes ({mem_kb} KB)     |"
    )


def _build_trav_table(m):
    """Build the Traversal sub-table for the Detailed Breakdown."""
    n_min = _fmt(m, "min_node_tests", fmt_str=".1f")
    n_max = _fmt(m, "max_node_tests", fmt_str=".1f")
    n_mean = _fmt(m, "mean_node_tests", fmt_str=".2f")
    t_min = _fmt(m, "min_tri_tests", fmt_str=".1f")
    t_max = _fmt(m, "max_tri_tests", fmt_str=".1f")
    t_mean = _fmt(m, "mean_tri_tests", fmt_str=".2f")
    s_min = _fmt(m, "min_shadow_tests", fmt_str=".1f")
    s_max = _fmt(m, "max_shadow_tests", fmt_str=".1f")
    s_mean = _fmt(m, "mean_shadow_tests", fmt_str=".2f")
    tr_min = _fmt(m, "min_traverse_tests", fmt_str=".1f")
    tr_max = _fmt(m, "max_traverse_tests", fmt_str=".1f")
    tr_mean = _fmt(m, "mean_traverse_tests", fmt_str=".2f")
    qd_min = _fmt(m, "min_query_depth", fmt_str=".1f")
    qd_max = _fmt(m, "max_query_depth", fmt_str=".1f")
    qd_mean = _fmt(m, "mean_query_depth", fmt_str=".2f")
    return (
        f"| node_tests     | {n_min:<6} | {n_max:<7} | {n_mean:<6} |\n"
        f"| tri_tests      | {t_min:<6} | {t_max:<7} | {t_mean:<6} |\n"
        f"| shadow_tests   | {s_min:<6} | {s_max:<7} | {s_mean:<6} |\n"
        f"| traverse_tests | {tr_min:<6} | {tr_max:<7} | {tr_mean:<6} |\n"
        f"| query_depth    | {qd_min:<6} | {qd_max:<7} | {qd_mean:<6} |"
    )


def get_device_info():
    """Detect GPU name and CPU core count."""
    gpu = "unknown"
    cores = 1
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=gpu_name", "--format=csv", "-i", "0"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            gpu = result.stdout.strip().split("\n")[-1].strip()
    except Exception:
        pass

    try:
        cores = os.cpu_count() or 1
    except Exception:
        pass

    return f"{gpu} / {cores} CPU cores"


def update_readme(results, readme_path):
    """Replace placeholders in README.md with actual metrics."""
    # Build lookup: (scene, config) -> metrics
    by_key = {}
    for scene, config, m in results:
        by_key[(scene, config)] = m

    with open(readme_path, "r") as f:
        lines = f.readlines()

    device = get_device_info()

    def _replace_section_table(heading, row_fn):
        """Find 'heading', ensure header+separator exist, then insert/replace data rows."""
        i = 0
        while i < len(lines):
            line = lines[i].rstrip("\n")
            if line == heading:
                # next non-blank line should be the header row
                hdr_idx = i + 1
                while hdr_idx < len(lines) and not lines[hdr_idx].strip().startswith(
                    "|"
                ):
                    hdr_idx += 1
                if hdr_idx >= len(lines):
                    return

                # check if separator row exists right after header
                sep_idx = hdr_idx + 1
                has_sep = sep_idx < len(lines) and "---" in lines[sep_idx]

                # Insert separator after header if it was missing (DO THIS FIRST)
                if not has_sep:
                    header = lines[hdr_idx].strip()
                    # Count cells: markdown row starts/ends with |, so cells = pipe_count - 1
                    num_cells = header.count("|") - 1
                    sep = "| " + " | ".join(["---"] * num_cells) + " |"
                    lines.insert(hdr_idx + 1, sep + "\n")

                # data starts at header + 2 (skip header and separator)
                data_start = hdr_idx + 2
                # find end of data rows (stop at blank line or non-table line)
                j = data_start
                while j < len(lines) and lines[j].strip().startswith("|"):
                    j += 1

                # ALWAYS insert data rows
                new_lines = []
                for scene, config, metrics in results:
                    new_lines.append(row_fn(scene, config, metrics) + "\n")
                lines[data_start:j] = new_lines
                return
            i += 1

    # --- Handle main BVH Performance summary table specially ---
    # This table is under "## BVH Performance" with no sub-heading.
    # Find the long header row and replace data rows after it.
    i = 0
    while i < len(lines):
        if "## BVH Performance" in lines[i]:
            j = i + 1
            # skip to header row (line starting with | and containing "Scene" and "Construction")
            while j < len(lines) and not (
                lines[j].strip().startswith("|")
                and "Construction" in lines[j]
                and "node_tests" in lines[j]
            ):
                j += 1
            if j < len(lines):
                j += 1  # skip header row
                if j < len(lines) and "---" in lines[j]:
                    j += 1  # skip separator row
                data_start = j
                # skip data rows
                while j < len(lines) and lines[j].strip().startswith("|"):
                    j += 1
                new_lines = []
                for scene, config, metrics in results:
                    new_lines.append(_summary_row(scene, config, metrics) + "\n")
                lines[data_start:j] = new_lines
            break
        i += 1

    # Insert device info after "Generated from" line
    for i, line in enumerate(lines):
        if "Generated from" in line:
            lines[i : i + 1] = [f"> Device: {device}\n", "\n"]
            break

    # Check which sections exist
    has_constr = any(line.strip() == "### Construction Metrics" for line in lines)
    has_trav = any(line.strip() == "### Traversal Metrics" for line in lines)

    _replace_section_table("### Construction Metrics", _constr_row)
    _replace_section_table("### Traversal Metrics", _trav_row)

    # If sections are missing, insert them before the first "### " heading after
    # "## BVH Performance" (or wherever the first "### " is).
    if not has_constr or not has_trav:
        # Find insertion point: after the summary table, before "## BVH Performance" or first "### "
        # Look for "### Detailed Per-Metric Breakdown" as anchor
        anchor = -1
        for i, line in enumerate(lines):
            if "### Detailed Per-Metric Breakdown" in line:
                anchor = i
                break
        if anchor == -1:
            anchor = len(lines)

        insert_lines = []
        if not has_constr:
            insert_lines.append("### Construction Metrics\n\n")
            insert_lines.append(
                "| Scene       | Config       | Const (s) | Nodes | Internal | "
                "Leaves | Leaf Depth (min/max/mean) | Prims/leaf (min/max/mean) |\n"
            )
            insert_lines.append(
                "| :---------- | :----------- | --------: | ----: | "
                "-------: | -----: | :----------------------- | :---------------------- |\n"
            )
            for scene, config, metrics in results:
                insert_lines.append(_constr_row(scene, config, metrics) + "\n")
            insert_lines.append("\n")

        if not has_trav:
            insert_lines.append("### Traversal Metrics\n\n")
            insert_lines.append(
                "| Scene       | Config       | Hit % | node_tests | tri_tests | "
                "shadow_tests | traverse_tests | query_depth | query_time |\n"
            )
            insert_lines.append(
                "| :---------- | :----------- | ----: | ---------: | "
                "--------: | -----------: | -------------: | ----------: | -----------: |\n"
            )
            for scene, config, metrics in results:
                insert_lines.append(_trav_row(scene, config, metrics) + "\n")
            insert_lines.append("\n")

        lines[anchor:anchor] = insert_lines

    def _find_section(marker):
        """Find line index of exact marker text."""
        for i, line in enumerate(lines):
            if line.strip() == marker:
                return i
        return -1

    def _find_next_heading(start_from):
        """Find next ##### or ### General heading."""
        for i in range(start_from + 1, len(lines)):
            s = lines[i].strip()
            if s.startswith("##### ") or (s.startswith("### ") and "General" in s):
                return i
        return len(lines)

    def _replace_block_table(block_start, block_end, search_for, build_fn):
        """Find a **bold** line, then replace following |rows| until blank."""
        idx = -1
        for i in range(block_start, min(block_end, len(lines))):
            if search_for in lines[i]:
                idx = i
                break
        if idx == -1:
            return
        t_start = idx + 1
        while t_start < block_end and not lines[t_start].strip().startswith("|"):
            t_start += 1
        t_end = t_start
        while t_end < block_end and lines[t_end].strip().startswith("|"):
            t_end += 1
        replacement = [nl + "\n" for nl in build_fn().split("\n") if nl]
        lines[t_start:t_end] = replacement

    # --- Construction sub-tables in Detailed Breakdown ---
    for scene, config, _ in results:
        pos = _find_section(f"##### {scene} - {config}")
        if pos == -1:
            continue
        next_pos = _find_next_heading(pos)
        _replace_block_table(
            pos,
            next_pos,
            "**Construction:**",
            lambda m=by_key[(scene, config)]: _build_constr_table(m),
        )

    # --- Traversal sub-tables in Detailed Breakdown ---
    for scene, config, _ in results:
        pos = _find_section(f"##### {scene} - {config}")
        if pos == -1:
            continue
        next_pos = _find_next_heading(pos)
        _replace_block_table(
            pos,
            next_pos,
            "**Traversal (all pixels):**",
            lambda m=by_key[(scene, config)]: _build_trav_table(m),
        )

    full = "".join(lines)
    full = full.replace(
        "\nThe `—` placeholders will be filled by `tests/bvh_metrics_unified.py` on the next run. "
        "This table shows the BVH tree structure: total node count, split between internal and "
        "leaf nodes, leaf depth range, and primitives stored per leaf node.",
        "",
    )
    full = full.replace(
        "\n> The `—` placeholders for construction and traversal details will be filled by "
        "`tests/bvh_metrics_unified.py` on the next run.\n",
        "",
    )
    with open(readme_path, "w") as f:
        f.write(full)

    print(f"\n  Updated {readme_path}")


def main():
    """Run all BVH metrics tests."""
    parser = argparse.ArgumentParser(description="BVH Metrics Test Script")
    parser.add_argument(
        "--write-readme",
        action="store_true",
        help="Update README.md tables with parsed metrics from existing runs",
    )
    args = parser.parse_args()

    os.makedirs(TESTS_DIR, exist_ok=True)

    # If --write-readme is passed, load existing results and update README
    if args.write_readme:
        print("=" * 70)
        print("  README UPDATE MODE (no raytracer runs)")
        print("=" * 70)
        print(f"Loading results from: {TESTS_DIR}")
        print()

        results = []
        for scene_name in SCENES:
            for config_name, _, _ in CONFIGS:
                result_file = os.path.join(TESTS_DIR, f"{scene_name}_{config_name}.txt")
                if os.path.exists(result_file):
                    with open(result_file, "r") as f:
                        output = f.read()
                    metrics = parse_metrics(output)
                    metrics["scene"] = scene_name
                    metrics["config"] = config_name
                    results.append((scene_name, config_name, metrics))
                    print(f"  Loaded: {scene_name} - {config_name}")
                else:
                    print(f"  Missing: {scene_name} - {config_name} (skipping)")

        if not results:
            print("\n  No results found. Run tests first without --write-readme.")
            sys.exit(1)

        readme_path = os.path.join(project_root, "README.md")
        update_readme(results, readme_path)

        return

    print("Warning: this could take around 10 mins to run all tests.")
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

    # Update README with new results
    readme_path = os.path.join(project_root, "README.md")
    update_readme(all_results, readme_path)


if __name__ == "__main__":
    main()
