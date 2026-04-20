"""Constants for project."""

DEVICE = "gpu"  # "cpu" or "gpu"

# for CPU python run:
CPU_DIMENSION = 500
# for GPU runs:
# 3070 mobile (local):
GPU_DIMENSION = 1440
# GPU_DIMENSION = 800
# # A100 (cluster):
# GPU_DIMENSION = 5760

USE_BVH_CACHE = True
MAX_BOUNCES = 16  # more than 32 is overkill, but counting on attenuation
IMG_FORMAT = "jpg"
SAMPLES = 16
DENOISE = False
RENDER_NON_BVH_STATS = False
PRINT_STATS = False
