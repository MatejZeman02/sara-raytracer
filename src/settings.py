"""Settings of the project."""

import os


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw_l = raw.strip().lower()
    if raw_l in ("1", "true", "yes", "on"):
        return True
    if raw_l in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"Invalid boolean value for {name}: '{raw}'")


# cpu-sequential, cpu-parallel, gpu
EXECUTION_MODE = os.getenv("RT_EXECUTION_MODE", "gpu").strip().lower()

if EXECUTION_MODE in ("cpu-sequential", "cpu-seq"):
    DEVICE = "cpu"
    CPU_PARALLEL = False
    EXECUTION_MODE = "cpu-sequential"
elif EXECUTION_MODE in ("cpu-parallel", "cpu-par", "cpu"):
    DEVICE = "cpu"
    CPU_PARALLEL = True
    EXECUTION_MODE = "cpu-parallel"
elif EXECUTION_MODE in ("gpu", "cuda"):
    DEVICE = "gpu"
    CPU_PARALLEL = False
    EXECUTION_MODE = "gpu"
else:
    raise ValueError(
        f"Unsupported RT_EXECUTION_MODE='{EXECUTION_MODE}'. "
        "Use one of: cpu-sequential, cpu-parallel, gpu"
    )

GPU_BLOCK_X = int(os.getenv("RT_GPU_BLOCK_X", "16"))
GPU_BLOCK_Y = int(os.getenv("RT_GPU_BLOCK_Y", str(GPU_BLOCK_X)))
if GPU_BLOCK_X <= 0 or GPU_BLOCK_Y <= 0:
    raise ValueError(
        f"GPU block dimensions must be positive, got ({GPU_BLOCK_X}, {GPU_BLOCK_Y})"
    )

EXECUTION_CONFIG = {
    "mode": EXECUTION_MODE,
    "device": DEVICE,
    "cpu_parallel": CPU_PARALLEL,
    "gpu_block": (GPU_BLOCK_X, GPU_BLOCK_Y),
}

WAVEFRONT_ENABLED = _env_bool("RT_WAVEFRONT_ENABLED", True)
BVH_OPS_BUDGET = int(os.getenv("RT_BVH_OPS_BUDGET", "500"))
if BVH_OPS_BUDGET <= 0 and WAVEFRONT_ENABLED:
    raise ValueError(f"RT_BVH_OPS_BUDGET must be > 0, got {BVH_OPS_BUDGET}")

WAVEFRONT_SORTING = _env_bool("RT_WAVEFRONT_SORTING", True)
_raw_sort_metric = os.getenv("RT_WAVEFRONT_SORT_METRIC", "ray_dir").strip().lower()
if _raw_sort_metric in ("material", "material_id", "mat", "mat_id"):
    WAVEFRONT_SORT_METRIC = "material"
elif _raw_sort_metric in ("ray_dir", "direction", "dir"):
    WAVEFRONT_SORT_METRIC = "ray_dir"
else:
    raise ValueError(
        "RT_WAVEFRONT_SORT_METRIC must be 'material' or 'ray_dir', "
        f"got '{_raw_sort_metric}'"
    )

WAVEFRONT_SORT_BACKEND = os.getenv("RT_WAVEFRONT_SORT_BACKEND", "auto").strip().lower()
if WAVEFRONT_SORT_BACKEND not in ("auto", "numpy", "cupy"):
    raise ValueError(
        "RT_WAVEFRONT_SORT_BACKEND must be auto, numpy, or cupy, "
        f"got '{WAVEFRONT_SORT_BACKEND}'"
    )

WAVEFRONT_SORT_DIR_BITS = int(os.getenv("RT_WF_SORT_DIR_BITS", "10"))
if WAVEFRONT_SORT_DIR_BITS < 4 or WAVEFRONT_SORT_DIR_BITS > 10:
    raise ValueError(
        "RT_WF_SORT_DIR_BITS must be between 4 and 10, "
        f"got {WAVEFRONT_SORT_DIR_BITS}"
    )

# for CPU njit python run:
CPU_DIMENSION = 800
# for GPU runs:
GPU_DIMENSION = 800
# GPU_DIMENSION = 5760

SCENE_NAME = os.getenv("RT_SCENE_NAME", "bunny").strip()
SAMPLES = 16
DENOISE = _env_bool("RT_DENOISE", True)
MAX_BOUNCES = 16
# more than 32 is overkill, but counting on attenuation
IMG_FORMAT = "jpg"
USE_BVH_CACHE = _env_bool("RT_USE_BVH_CACHE", True)
PRINT_STATS = True
RENDER_NON_BVH_STATS = False
