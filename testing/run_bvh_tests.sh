#!/bin/bash
# Run BVH metrics tests with different configurations
# Usage: ./testing/run_bvh_tests.sh

set -e

CONDA_RUN="/home/bubakulus/miniforge3/bin/conda run -n raytracer python"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." &> /dev/null && pwd)"
OUTPUT_DIR="$PROJECT_ROOT/src/output"
RESULTS_FILE="$OUTPUT_DIR/bvh_test_results.txt"

mkdir -p "$OUTPUT_DIR"

echo "================================================================="
echo "  BVH METRICS TEST SUITE"
echo "================================================================="
echo ""
echo "Results will be saved to: $RESULTS_FILE"
echo ""

# Function to run a test with given settings
run_test() {
    local test_name="$1"
    local scene="$2"
    local dimension="$3"
    local collect_stats="$4"
    local render_non_bvh="$5"
    local denoise="$6"
    local max_bounces="$7"
    local samples="$8"
    local tonemapper="$9"

    echo "-----------------------------------------------------------------"
    echo "  TEST: $test_name"
    echo "-----------------------------------------------------------------"
    echo "  Scene: $scene"
    echo "  Resolution: ${dimension}x${dimension}"
    echo "  BVH Stats: $collect_stats"
    echo "  Non-BVH: $render_non_bvh"
    echo "  Denoise: $denoise"
    echo "  Max Bounces: $max_bounces"
    echo "  Samples: $samples"
    echo "  Tonemapper: $tonemapper"
    echo ""

    # Write settings to src/settings.py
    cat > "$PROJECT_ROOT/src/settings.py" <<EOF
DEVICE = "gpu"
CPU_DIMENSION = 500
GPU_DIMENSION = $dimension
SCENE_NAME = "$scene"
SAMPLES = $samples
DENOISE = $denoise
MAX_BOUNCES = $max_bounces
IMG_FORMAT = "jpg"
USE_BVH_CACHE = True
PRINT_STATS = False
RENDER_NON_BVH_STATS = $render_non_bvh
TONEMAPPER = "$tonemapper"
COLLECT_BVH_STATS = $collect_stats
EOF

    # Run the test
    local start_time=$(date +%s%N)
    local output=$($CONDA_RUN -c conda-forge python -c "
import sys
sys.argv.append('--collect-bvh-stats')
from src import main
main()
" 2>&1)
    local end_time=$(date +%s%N)
    local duration=$(( (end_time - start_time) / 1000000 ))

    echo "  Duration: ${duration}ms"
    echo ""

    # Extract key metrics from output
    local hit_pixels=$(echo "$output" | grep "Hit pixels:" | head -1 | sed 's/.*Hit pixels:\s*//' | sed 's/ .*//')
    local miss_pixels=$(echo "$output" | grep "Miss pixels:" | head -1 | sed 's/.*Miss pixels:\s*//' | sed 's/ .*//')
    local mean_node=$(echo "$output" | grep "Overall mean node_tests:" | head -1 | sed 's/.*: *//')
    local mean_tri=$(echo "$output" | grep "Overall mean tri_tests:" | head -1 | sed 's/.*: *//')
    local mean_shadow=$(echo "$output" | grep "Overall mean shadow_tests:" | head -1 | sed 's/.*: *//')
    local hit_mean_node=$(echo "$output" | grep "Hit-pixel mean node_tests:" | head -1 | sed 's/.*: *//')
    local hit_mean_tri=$(echo "$output" | grep "Hit-pixel mean tri_tests:" | head -1 | sed 's/.*: *//')
    local hit_mean_shadow=$(echo "$output" | grep "Hit-pixel mean shadow_tests:" | head -1 | sed 's/.*: *//')

    echo "  Results:"
    echo "    Hit pixels: $hit_pixels"
    echo "    Mean node_tests (all): $mean_node"
    echo "    Mean tri_tests (all): $mean_tri"
    echo "    Mean shadow_tests (all): $mean_shadow"
    echo "    Hit-pixel mean node_tests: $hit_mean_node"
    echo "    Hit-pixel mean tri_tests: $hit_mean_tri"
    echo "    Hit-pixel mean shadow_tests: $hit_mean_shadow"
    echo ""

    # Save to results file
    cat >> "$RESULTS_FILE" <<EOF
=== $test_name ===
Scene: $scene
Resolution: ${dimension}x${dimension}
BVH Stats: $collect_stats
Non-BVH: $render_non_bvh
Denoise: $denoise
Max Bounces: $max_bounces
Samples: $samples
Tonemapper: $tonemapper
Duration: ${duration}ms
Hit pixels: $hit_pixels
Miss pixels: $miss_pixels
Mean node_tests (all): $mean_node
Mean tri_tests (all): $mean_tri
Mean shadow_tests (all): $mean_shadow
Hit-pixel mean node_tests: $hit_mean_node
Hit-pixel mean tri_tests: $hit_mean_tri
Hit-pixel mean shadow_tests: $hit_mean_shadow

EOF

    echo ""
}

# Test 1: Bunny scene with BVH (default)
run_test "Bunny + BVH (GPU)" "bunny" 1024 True False True 16 16 "none"

# Test 2: Bunny scene without BVH (brute force)
run_test "Bunny + No BVH (GPU)" "bunny" 512 False False True 16 16 "none"

# Test 3: Box-spheres with BVH
run_test "Box-Spheres + BVH (GPU)" "box-spheres" 1024 True False True 16 16 "none"

# Test 4: Box-spheres without BVH
run_test "Box-Spheres + No BVH (GPU)" "box-spheres" 512 False False True 16 16 "none"

# Test 5: Bunny with BVH, 1 bounce
run_test "Bunny + BVH (1 bounce)" "bunny" 1024 True False True 1 16 "none"

# Test 6: Bunny with BVH, 4 bounces
run_test "Bunny + BVH (4 bounces)" "bunny" 1024 True False True 4 16 "none"

# Test 7: Bunny with BVH, 8 bounces
run_test "Bunny + BVH (8 bounces)" "bunny" 1024 True False True 8 16 "none"

echo "================================================================="
echo "  ALL TESTS COMPLETE"
echo "================================================================="
echo ""
echo "Results saved to: $RESULTS_FILE"
echo ""
echo "To view results:"
echo "  cat $RESULTS_FILE"
echo ""
