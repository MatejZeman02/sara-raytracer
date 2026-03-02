"""Constants for renderer."""
from numpy import float32

CULLBACK = True
EPSILON = float32(1e-4)  # dynamic to size of the scene
BIG_EPSILON = float32(1e-2)
INFINITY_VEC = (float32(-1.0), float32(0.0), float32(0.0))
DIMENSION = 1440  # 3070 mobile (local)
# DIMENSION = 800  # 3070 mobile (local)
# DIMENSION = 5760 # A100 (cluster)
BLOCK_THREADS = 16
RENDER_NON_BVH_STATS = False
USE_CACHE = True
STACK_SIZE = 64
MAX_BOUNCES = 160
