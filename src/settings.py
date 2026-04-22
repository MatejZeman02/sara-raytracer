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
