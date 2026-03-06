"""Constants for project."""

DEVICE = "cpu"  # "cpu" or "gpu"

# for CPU python run:
CPU_DIMENSION = 1440
# for GPU runs:
# 3070 mobile (local):
GPU_DIMENSION = 1440
# GPU_DIMENSION = 800
# # A100 (cluster):
# GPU_DIMENSION = 5760

RENDER_NON_BVH_STATS = True
USE_BVH_CACHE = True
MAX_BOUNCES = 16  # more than 32 is overkill, but counting on attenuation
