import numpy as np
import src.settings as settings
settings.DEVICE = "cpu"
from src.bvh import build_bvh

triangles = np.array([
    [[0,0,0], [1,0,0], [0,1,0]],
    [[1,0,0], [1,1,0], [0,1,0]]
], dtype=np.float32)

tri_normals = np.zeros((2,3,3), dtype=np.float32)
tri_uvs = np.zeros((2,3,2), dtype=np.float32)
mat_indices = np.zeros(2, dtype=np.int32)
nodes, tris, norms, uvs, mats = build_bvh(triangles, tri_normals, tri_uvs, mat_indices)
print(nodes)
