import sys, os; sys.path.append(os.path.join(os.getcwd(), 'src'))
import numpy as np
from numpy import float32

light_data = {"pos": [2.75, 4.90, 2.75], "color": [1, 1, 1]}
cam_data = {"pos": [278, 273, -800], "up": [0, 1, 0], "dir": [0, 0, 1], "fov": 0.68}
width, height = 1440, 1440

fov = float32(cam_data["fov"])
dir_vec = np.array(cam_data["dir"], dtype=float32)
up_vec = np.array(cam_data["up"], dtype=float32)

b_vec = np.cross(dir_vec, up_vec)
b_vec = b_vec / np.linalg.norm(b_vec)

t = float32(1.0)
g_w = float32(2.0) * t * np.tan(fov / float32(2.0))
g_h = g_w * (height / width)

q_w = (g_w / (width - 1)) * b_vec
q_h = (g_h / (height - 1)) * up_vec
p00 = t * dir_vec - (g_w / float32(2.0)) * b_vec + (g_h / float32(2.0)) * up_vec

print(f"p00:    {p00}")
print(f"q_w:    {q_w}")
print(f"q_h:    {q_h}")

center_kx, center_ky = width/2, height/2
ray_dir = p00 + q_w * center_kx + q_h * center_ky
ray_dir = ray_dir / np.linalg.norm(ray_dir)
print(f"Center Ray: {ray_dir}")
