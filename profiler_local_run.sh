#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_ROOT="$PROJECT_ROOT/benchmark_logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$LOG_ROOT/ncu_${STAMP}"
mkdir -p "$OUT_DIR"

OPEN_UI=0
if [[ "${1:-}" == "--ui" ]]; then
	OPEN_UI=1
fi

SCENE="${RT_SCENE_NAME:-dragon}"
MODE="${RT_EXECUTION_MODE:-gpu}"
BLOCK_X="${RT_GPU_BLOCK_X:-16}"
BLOCK_Y="${RT_GPU_BLOCK_Y:-16}"
REPORT_BASE="$OUT_DIR/profile_report"

NCU_BIN=""
if command -v ncu >/dev/null 2>&1; then
	NCU_BIN="$(command -v ncu)"
else
	NCU_BIN="$(conda run -n raytracer which ncu 2>/dev/null | awk '/^\//{print; exit}')"
fi

if [[ -z "$NCU_BIN" ]]; then
	echo "ERROR: Nsight Compute CLI (ncu) not found."
	echo "Install ncu or ensure it is available in PATH/conda env 'raytracer'."
	exit 127
fi

NCU_UI_BIN="$(dirname "$NCU_BIN")/ncu-ui"

echo "Profiling with Nsight Compute"
echo "  scene      : $SCENE"
echo "  mode       : $MODE"
echo "  block size : ${BLOCK_X}x${BLOCK_Y}"
echo "  output dir : $OUT_DIR"
echo "  ncu        : $NCU_BIN"

# Capture warp-state related sections and keep CLI logs for report evidence.
PROFILE_RC=0
if RT_SCENE_NAME="$SCENE" \
	RT_USE_BVH_CACHE=1 \
	RT_DENOISE=0 \
	"$NCU_BIN" -f \
		--set full \
		--section WarpStateStats \
		--section SchedulerStats \
		--section LaunchStats \
		-o "$REPORT_BASE" \
		conda run -n raytracer python "$PROJECT_ROOT/src/main.py" \
			--mode "$MODE" \
			--block-x "$BLOCK_X" \
			--block-y "$BLOCK_Y" \
		> "$OUT_DIR/ncu_run.log" 2>&1; then
	PROFILE_RC=0
else
	PROFILE_RC="$?"
fi

if [[ "$PROFILE_RC" -ne 0 ]]; then
	echo "Profiler run failed with exit code $PROFILE_RC"
	echo "  log    : $OUT_DIR/ncu_run.log"

	if grep -q "ERR_NVGPUCTRPERM" "$OUT_DIR/ncu_run.log"; then
		echo "Detected missing permission for NVIDIA GPU performance counters."
		echo "Enable counters for non-root users (e.g., NVreg_RestrictProfilingToAdminUsers=0)"
		echo "or run profiling with sufficient privileges."
	fi

	exit "$PROFILE_RC"
fi

# Best-effort export of warp state data to CSV/text for non-UI analysis.
if [[ -f "${REPORT_BASE}.ncu-rep" ]]; then
	"$NCU_BIN" --import "${REPORT_BASE}.ncu-rep" --page details --section WarpStateStats --csv \
		> "$OUT_DIR/ncu_warpstate.csv" 2>/dev/null || true
	"$NCU_BIN" --import "${REPORT_BASE}.ncu-rep" --page details --section WarpStateStats \
		> "$OUT_DIR/ncu_warpstate.txt" 2>/dev/null || true
else
	echo "No .ncu-rep report generated; skipping warp-state export files."
fi

echo "Profiler run completed."
echo "  report : ${REPORT_BASE}.ncu-rep"
echo "  log    : $OUT_DIR/ncu_run.log"
echo "  warp csv/txt: $OUT_DIR/ncu_warpstate.csv , $OUT_DIR/ncu_warpstate.txt"

if [[ "$OPEN_UI" -eq 1 ]]; then
	if [[ -x "$NCU_UI_BIN" ]]; then
		"$NCU_UI_BIN" "${REPORT_BASE}.ncu-rep"
	else
		echo "ncu-ui not found next to $NCU_BIN"
	fi
fi
