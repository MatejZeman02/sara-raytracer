"""Constants for kernels."""

import math
from numpy import float32, uint8, random

# epsilon is dynamic to size of the scene
# used exclusively for the `det` check in intersect_triangle (strict tolerance)
DET_EPSILON = float32(1e-9)
# used for t-distance checks and shadow ray origin offsets (scene-scale tolerance)
RAY_EPSILON = float32(1e-5)
DIST_TO_LIGHT_MULT = float32(1e-3)  # t max safeguard for shadow rays

BARYCENTRIC_EPSILON = float32(1e-7)  # acounts with float32 precision error
DENOMINATOR_EPSILON = float32(1e-10)  # division by zero and its check statements

THROUGHPUT_THRESHOLD = float32(1e-2)  # stop tracing rays where addition is negligible
INFINITY_VEC = (float32(-1.0), float32(0.0), float32(0.0))
BLOCK_THREADS = 16
STACK_SIZE = 64
SEED = random.randint(0, 2**32 - 1)  # or 42

# Material property indices (columns in materials array)
MAT_DIFFUSE_R = 0
MAT_DIFFUSE_G = 1
MAT_DIFFUSE_B = 2
MAT_SPECULAR_R = 3
MAT_SPECULAR_G = 4
MAT_SPECULAR_B = 5
MAT_ROUGHNESS = 6
MAT_EMISSIVE_R = 7
MAT_EMISSIVE_G = 8
MAT_EMISSIVE_B = 9
MAT_TRANSMISSION_R = 10
MAT_TRANSMISSION_G = 11
MAT_TRANSMISSION_B = 12
MAT_IOR = 13
NO_TEXTURE = -1

# BVH node layout indices (columns in bvh_nodes array)
BVH_MIN_X = 0
BVH_MIN_Y = 1
BVH_MIN_Z = 2
BVH_MAX_X = 3
BVH_MAX_Y = 4
BVH_MAX_Z = 5
BVH_LEFT_OR_START = 6
BVH_RIGHT_OR_COUNT = 7

# 32 bits float constants (It's almost like in OpenGl):
ZERO = float32(0.0)
HALF = float32(0.5)
ONE = float32(1.0)
TWO = float32(2.0)
PI = float32(math.pi)

UINT8_MAX_F = float32(255.0)
UINT8_MAX_I = uint8(255)

MISS_GRAY_F = float32(0.0)  # background color (in linear space)
MISS_GRAY = MISS_GRAY_F / UINT8_MAX_F
MISS_COLOR_F = (MISS_GRAY, MISS_GRAY, MISS_GRAY)

# out stats layout:
# 0: primary tri tests
PRIMARY_TRI = 0
# 1: primary node tests
PRIMARY_NODE = 1
# 2: primary rays
PRIMARY_RAY = 2
# 3: secondary rays (refractions + bounces)
SECONDARY_RAY = 3
# 4: shadow rays
SHADOW_RAY = 4
# 5: max traversal depth (stack depth)
TRAVERSAL_DEPTH = 5
# 6: query time (only for CPU)
QUERY_TIME = 6
# 7: traverse tests
TRAVERSE_TESTS = 7
# 8: query depth
QUERY_DEPTH = 8

# gpu metrics array layout (per-pixel, shape: width*height, 4):
# 0: node_tests (float32) - number of bvh node visits
METRICS_NODE_TESTS = 0
# 1: tri_tests (float32) - number of triangle intersection tests
METRICS_TRI_TESTS = 1
# 2: shadow_tests (float32) - number of shadow ray tests
METRICS_SHADOW_TESTS = 2
# 3: is_hit (uint8, stored as float32 0/1) - whether ray hit geometry
METRICS_IS_HIT = 3
