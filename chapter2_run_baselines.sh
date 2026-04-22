#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_ROOT="$PROJECT_ROOT/benchmark_logs"
STAMP="${1:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="$LOG_ROOT/chapter2_${STAMP}"
CSV_FILE="$OUT_DIR/results.csv"
DRAGON_SCENE="${RT_DRAGON_SCENE:-dragon}"
RUN_CPU="${RT_BENCH_RUN_CPU:-1}"
RUN_GPU="${RT_BENCH_RUN_GPU:-1}"

mkdir -p "$OUT_DIR"

echo "case,mode,scene,block_x,block_y,status,render_time_s,total_rays,mrays_per_s,log_file" > "$CSV_FILE"

parse_metrics_to_csv() {
    local case_name="$1"
    local mode="$2"
    local scene="$3"
    local block_x="$4"
    local block_y="$5"
    local log_file="$6"
    local csv_file="$7"
    local status="$8"

    python - "$case_name" "$mode" "$scene" "$block_x" "$block_y" "$log_file" "$csv_file" "$status" <<'PY'
import csv
import pathlib
import re
import sys

case_name, mode, scene, block_x, block_y, log_file, csv_file, status = sys.argv[1:]
text = pathlib.Path(log_file).read_text(encoding="utf-8", errors="ignore")

times = re.findall(r"\[metrics\]\s+[^:\n]+:\s*([0-9]+(?:\.[0-9]+)?)\s*s", text)
render_time = times[-1] if times else ""

rays_match = re.search(r"\[metrics\]\s+total rays cast\s*:\s*([0-9,]+)", text)
total_rays = rays_match.group(1).replace(",", "") if rays_match else ""

through_match = re.search(r"\[metrics\]\s+throughput\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*MRays/s", text)
throughput = through_match.group(1) if through_match else ""

with open(csv_file, "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            case_name,
            mode,
            scene,
            block_x,
            block_y,
            status,
            render_time,
            total_rays,
            throughput,
            log_file,
        ]
    )
PY
}

run_case() {
    local case_name="$1"
    local mode="$2"
    local scene="$3"
    local block_x="$4"
    local block_y="$5"

    local log_file="$OUT_DIR/${case_name}.log"
    local block_label="${block_x}x${block_y}"
    if [[ "$mode" != "gpu" ]]; then
        block_label="n/a"
    fi

    echo "\n=== ${case_name} (mode=${mode}, scene=${scene}, block=${block_label}) ==="

    local -a cmd
    cmd=(conda run -n raytracer python "$PROJECT_ROOT/src/main.py" --mode "$mode")
    if [[ "$mode" == "gpu" ]]; then
        cmd+=(--block-x "$block_x" --block-y "$block_y")
    fi

    if RT_SCENE_NAME="$scene" \
        RT_USE_BVH_CACHE=1 \
        RT_DENOISE=0 \
        "${cmd[@]}" \
            > "$log_file" 2>&1; then
        parse_metrics_to_csv "$case_name" "$mode" "$scene" "$block_x" "$block_y" "$log_file" "$CSV_FILE" "OK"
        grep "\[metrics\]" "$log_file" || true
    else
        local rc="$?"
        parse_metrics_to_csv "$case_name" "$mode" "$scene" "$block_x" "$block_y" "$log_file" "$CSV_FILE" "FAILED($rc)"
        echo "Case $case_name failed with exit code $rc"
        tail -n 30 "$log_file" || true
    fi
}

prewarm_scene_cache() {
    local scene="$1"
    local log_file="$OUT_DIR/prewarm_${scene}.log"

    echo "\n--- Prewarming cache for scene=${scene} ---"
    if RT_SCENE_NAME="$scene" \
        RT_USE_BVH_CACHE=1 \
        RT_DENOISE=0 \
        conda run -n raytracer python "$PROJECT_ROOT/src/main.py" \
            --mode gpu \
            --block-x 16 \
            --block-y 16 \
            > "$log_file" 2>&1; then
        echo "Prewarm for $scene completed."
    else
        echo "Prewarm for $scene failed; continuing benchmarks anyway."
        tail -n 30 "$log_file" || true
    fi
}

if [[ "$RUN_CPU" == "1" ]]; then
    prewarm_scene_cache "box-scaled"
fi
if [[ "$RUN_CPU" == "1" || "$RUN_GPU" == "1" ]]; then
    prewarm_scene_cache "$DRAGON_SCENE"
fi

if [[ "$RUN_CPU" == "1" ]]; then
    run_case "cpu_seq_box_scaled" "cpu-sequential" "box-scaled" "" ""
    run_case "cpu_seq_${DRAGON_SCENE}" "cpu-sequential" "$DRAGON_SCENE" "" ""
    run_case "cpu_par_box_scaled" "cpu-parallel" "box-scaled" "" ""
    run_case "cpu_par_${DRAGON_SCENE}" "cpu-parallel" "$DRAGON_SCENE" "" ""
fi

if [[ "$RUN_GPU" == "1" ]]; then
    run_case "gpu_${DRAGON_SCENE}_8x8" "gpu" "$DRAGON_SCENE" "8" "8"
    run_case "gpu_${DRAGON_SCENE}_16x16" "gpu" "$DRAGON_SCENE" "16" "16"
    run_case "gpu_${DRAGON_SCENE}_32x32" "gpu" "$DRAGON_SCENE" "32" "32"
fi

echo "\nBaseline benchmark completed."
echo "Output directory: $OUT_DIR"
echo "Results CSV: $CSV_FILE"
