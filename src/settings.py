"""Constants for project."""

DEVICE = "gpu"  # "cpu" or "gpu"

# for CPU python run:
CPU_DIMENSION = 500
# for GPU runs:
GPU_DIMENSION = 150
# GPU_DIMENSION = 5760

# SCENE_NAME = "box-advanced"

SCENE_NAME = "box-spheres"
SAMPLES = 16
DENOISE = False
MAX_BOUNCES = 16
# more than 32 is overkill, but counting on attenuation
IMG_FORMAT = "jpg"
USE_BVH_CACHE = True
PRINT_STATS = False
RENDER_NON_BVH_STATS = False
