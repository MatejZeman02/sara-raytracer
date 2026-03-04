"""Constants for project."""

DEVICE = "gpu"  # "cpu" or "gpu"
CULLBACK = True

# for CPU python run:
CPU_DIMENSION = 1440
# for GPU runs:
# 3070 mobile (local):
GPU_DIMENSION = 1440
# GPU_DIMENSION = 800
# # A100 (cluster):
# GPU_DIMENSION = 5760

RENDER_NON_BVH_STATS = False
USE_CACHE = True
MAX_BOUNCES = 160 # more than 160 is overkill, but counting on attenuation
