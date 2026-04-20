"""Constants for project."""

DEVICE = "gpu"  # "cpu" or "gpu"

# for CPU python run:
CPU_DIMENSION = 500
# for GPU runs:
# 3070 mobile (local):
# GPU_DIMENSION = 800
# # A100 (cluster):
# GPU_DIMENSION = 5760

# SCENE_NAME = "box-advanced"

GPU_DIMENSION = 1024
SCENE_NAME = "dragon"
SAMPLES = 6
DENOISE = True
MAX_BOUNCES = 16
# more than 32 is overkill, but counting on attenuation
IMG_FORMAT = "jpg"
USE_BVH_CACHE = False
PRINT_STATS = True
RENDER_NON_BVH_STATS = True
