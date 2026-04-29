import os
import subprocess
import datetime
import re
import csv

from pathlib import Path

import matplotlib.pyplot as plt
import numpy

PROJECT_ROOT = Path(__file__).parent.resolve()
SRC_MAIN = PROJECT_ROOT / "src" / "main.py"
stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# detecting gpu for folder naming (without numba)
try:
    smi_out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    )
    gpu_name = smi_out.strip().replace(" ", "_").lower()
except Exception:
    gpu_name = "unknown_gpu"

out_dir = PROJECT_ROOT / "benchmark_logs" / f"chapter2_{gpu_name}_{stamp}"
out_dir.mkdir(parents=True, exist_ok=True)
csv_file = out_dir / "results.csv"

# csv headers
headers = [
    "case",
    "mode",
    "scene",
    "block_x",
    "block_y",
    "status",
    "render_time_s",
    "total_rays",
    "mrays_per_s",
    "wavefront_enabled",
    "ops_budget",
    "pass1_time_s",
    "compaction_time_s",
    "pass2_time_s",
    "active_rays_compacted",
    "log_file",
]
with open(csv_file, "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow(headers)

# regexes:
p_time = re.compile(r"\[metrics\]\s+[^:\n]+:\s*([0-9]+(?:\.[0-9]+)?)\s*s")
p_rays = re.compile(r"\[metrics\]\s+total rays cast\s*:\s*([0-9,]+)")
p_thr = re.compile(r"\[metrics\]\s+throughput\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*MRays/s")
p_p1 = re.compile(r"\[timing\]\s+pass1 render\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*s")
p_cp = re.compile(r"\[timing\]\s+cpu compaction\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*s")
p_p2 = re.compile(r"\[timing\]\s+pass2 render\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*s")
p_ar = re.compile(r"\[timing\]\s+active rays after p1\s*:\s*([0-9,]+)")


def run_case(case_name, block_size, wf_en=0, budget=0):
    """run a single benchmark configuration"""
    log_path = out_dir / f"{case_name}.log"
    env = os.environ.copy()
    env.update(
        {
            "RT_SCENE_NAME": "dragon",
            "RT_USE_BVH_CACHE": "1",
            "RT_DENOISE": "0",
            "RT_WAVEFRONT_ENABLED": str(wf_en),
            "RT_BVH_OPS_BUDGET": str(budget),
            "RT_WAVEFRONT_SORT_BACKEND": "numpy",
            "RT_WAVEFRONT_SORT_METRIC": "material",
        }
    )

    cmd = [
        "python",
        str(SRC_MAIN),
        "--mode",
        "gpu",
        "--block-x",
        str(block_size),
        "--block-y",
        str(block_size),
    ]

    print(
        f"running {case_name} (block: {block_size}x{block_size}, wf: {wf_en}, budget: {budget})..."
    )

    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    log_path.write_text(result.stdout + "\n" + result.stderr)

    status = "OK" if result.returncode == 0 else f"FAILED({result.returncode})"
    text = result.stdout

    times = p_time.findall(text)
    rays = p_rays.search(text)
    thr = p_thr.search(text)
    p1 = p_p1.search(text)
    cp = p_cp.search(text)
    p2 = p_p2.search(text)
    ar = p_ar.search(text)

    row = [
        case_name,
        "gpu",
        "dragon",
        str(block_size),
        str(block_size),
        status,
        times[-1] if times else "",
        rays.group(1).replace(",", "") if rays else "",
        thr.group(1) if thr else "",
        str(wf_en),
        str(budget),
        p1.group(1) if p1 else "0",
        cp.group(1) if cp else "0",
        p2.group(1) if p2 else "0",
        ar.group(1).replace(",", "") if ar else "0",
        str(log_path),
    ]

    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def plot_results():
    """generate comparison plot"""
    budgets = []
    p1_times = []
    comp_times = []
    p2_times = []
    tot_times = []
    mega_baseline = 0.0

    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)

        # collect data in a list to sort it later
        wf_data = []
        for row in reader:
            if row["status"] != "OK":
                continue
            if row["wavefront_enabled"] == "0" and row["block_x"] == "16":
                mega_baseline = float(row["render_time_s"])
            elif row["wavefront_enabled"] == "1":
                wf_data.append(
                    {
                        "budget": int(row["ops_budget"]),
                        "p1": float(row["pass1_time_s"]),
                        "comp": float(row["compaction_time_s"]),
                        "p2": float(row["pass2_time_s"]),
                        "tot": float(row["render_time_s"]),
                    }
                )

    if not wf_data:
        print("no wavefront data found to plot")
        return

    # sort by budget to prevent crisscrossing interpolation lines
    wf_data.sort(key=lambda x: x["budget"])

    budgets = [d["budget"] for d in wf_data]
    p1_times = [d["p1"] for d in wf_data]
    comp_times = [d["comp"] for d in wf_data]
    p2_times = [d["p2"] for d in wf_data]
    tot_times = [d["tot"] for d in wf_data]

    _fig, ax = plt.subplots(figsize=(10, 6))

    # change x axis to logarithmic scale to handle the wide spread of budget values
    ax.set_xscale("log")

    # force x axis ticks to match our exact tested budgets
    ax.set_xticks(budgets)
    ax.set_xticklabels([str(b) for b in budgets], rotation=45)

    # stackplot for wavefront memory overhead and compute
    ax.stackplot(
        budgets,
        p1_times,
        comp_times,
        p2_times,
        labels=["wavefront: pass 1", "wavefront: compaction", "wavefront: pass 2"],
        colors=["#a1c9f4", "#ffb482", "#8de5a1"],
        alpha=0.9,
    )

    # line plot for total wavefront time
    ax.plot(
        budgets,
        tot_times,
        color="black",
        linewidth=2,
        marker="o",
        label="wavefront: total time",
    )

    # render megakernel baseline as a dashed line and a shaded target area
    if mega_baseline > 0:
        ax.axhline(
            mega_baseline,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"megakernel baseline: {mega_baseline:.3f}s",
        )
        ax.fill_between(
            budgets,
            0,
            mega_baseline,
            color="red",
            alpha=0.1,
            label="megakernel target zone",
        )

    ax.set_title(f"wavefront vs megakernel render time ({gpu_name})", fontsize=14)
    ax.set_xlabel("bvh operations budget (log scale)", fontsize=12)
    ax.set_ylabel("render time (s)", fontsize=12)

    # display grid for both major and minor ticks
    ax.grid(True, which="both", linestyle=":", alpha=0.7)
    ax.legend(loc="lower right")

    plt.tight_layout()
    out_png = out_dir / "wavefront_ucurve.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"graph saved to {out_png}")


############### MAIN ###############

if __name__ == "__main__":
    print(f"starting benchmark on {gpu_name}")
    print("prewarming jit...")
    run_case("prewarm", 16, 0, 9999)

    print("running megakernel baselines...")
    run_case("gpu_dragon_8x8", 8, 0, 99999)
    run_case("gpu_dragon_16x16", 16, 0, 99999)
    run_case("gpu_dragon_32x32", 32, 0, 99999)

    print("running wavefront budget sweep (16x16)...")
    for b in [16, 32, 64, 128, 256, 512, 1024, 2048, 4000, 6000, 8000, 10_000, 16000, 32000]:
        run_case(f"gpu_dragon_16x16_wf_{b}", 16, 1, b)

    plot_results()

    subprocess.run(f"nvidia-smi > {out_dir}/nvidia_smi.txt", shell=True)
    print(f"done. analysis is in {out_dir}")
