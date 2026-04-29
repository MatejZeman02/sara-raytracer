#!/usr/bin/env bash
set -uo pipefail

BUDGET=8000
SCENE="box-advanced"
OUT_DIR="benchmark_logs"
mkdir -p "$OUT_DIR"

REP_MEGA="$OUT_DIR/ncu_megakernel_profile"
REP_WF="$OUT_DIR/ncu_wavefront_pass2_profile"
KERNEL_REGEX="render_kernel_pass2" 

# capture absolute paths before invoking sudo so secure_path doesn't blind us
NCU_BIN="$(command -v ncu)"
PYTHON_BIN="$(command -v python)"

echo "Using NCU: $NCU_BIN"
echo "Using Python: $PYTHON_BIN"
echo ""

# helper function to extract the metric
extract_threads() {
    local rep_file="$1"
    "$NCU_BIN" --import "${rep_file}.ncu-rep" --page details --section WarpStateStats --csv 2>/dev/null \
        | grep "Avg. Active Threads Per Warp" \
        | tail -n 1 \
        | sed -n 's/.*,"\([^"]*\)",*$/\1/p'
}

echo "====================================================="
echo "1. Profiling Megakernel (Wavefront Disabled)"
echo "====================================================="
sudo --preserve-env=PATH,LD_LIBRARY_PATH,CONDA_PREFIX \
RT_SCENE_NAME=$SCENE \
RT_USE_BVH_CACHE=1 \
RT_DENOISE=0 \
RT_WAVEFRONT_ENABLED=0 \
RT_WAVEFRONT_SORT_BACKEND="numpy" \
RT_WAVEFRONT_SORT_METRIC="ray_dir" \
"$NCU_BIN" \
    --target-processes all \
    --set default \
    --section WarpStateStats \
    --export "$REP_MEGA" \
    --force \
    "$PYTHON_BIN" src/main.py --mode gpu --block-x 16 --block-y 16

echo ""
echo "====================================================="
echo "2. Profiling Wavefront Pass 2 (Budget: $BUDGET)"
echo "====================================================="
sudo --preserve-env=PATH,LD_LIBRARY_PATH,CONDA_PREFIX \
RT_SCENE_NAME=$SCENE \
RT_USE_BVH_CACHE=1 \
RT_DENOISE=0 \
RT_WAVEFRONT_ENABLED=1 \
RT_BVH_OPS_BUDGET=$BUDGET \
RT_WAVEFRONT_SORT_BACKEND="numpy" \
RT_WAVEFRONT_SORT_METRIC="ray_dir" \
"$NCU_BIN" \
    --target-processes all \
    --kernel-name regex:".*${KERNEL_REGEX}.*" \
    --set default \
    --section WarpStateStats \
    --export "$REP_WF" \
    --force \
    "$PYTHON_BIN" src/main.py --mode gpu --block-x 16 --block-y 16

echo ""
echo "====================================================="
echo "3. Warp Divergence Results"
echo "====================================================="

# set permissions so normal user can read the files
sudo chown -R "$USER:$USER" "$OUT_DIR"

MEGA_THREADS=$(extract_threads "$REP_MEGA")
WF_THREADS=$(extract_threads "$REP_WF")

echo "Megakernel Active Threads/Warp : ${MEGA_THREADS:-ERROR} / 32"
echo "Wavefront Active Threads/Warp  : ${WF_THREADS:-ERROR} / 32"
echo "====================================================="
