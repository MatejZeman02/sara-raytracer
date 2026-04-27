"""Constants for project."""

DEVICE = "gpu"  # "cpu" or "gpu"

# for CPU python run:
CPU_DIMENSION = 500
# for GPU runs:
GPU_DIMENSION = 1024
# GPU_DIMENSION = 5760

# SCENE_NAME = "box-advanced"

SCENE_NAME = "box-scaled"
SAMPLES = 16
DENOISE = True
MAX_BOUNCES = 16
# more than 32 is overkill, but counting on attenuation
IMG_FORMAT = "jpg"
USE_BVH_CACHE = False
PRINT_STATS = False
RENDER_NON_BVH_STATS = False
