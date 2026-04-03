import sys, os; sys.path.append(os.path.join(os.getcwd(), 'src'))
import numpy as np
from src.setup_vectors import build_setup_vectors

light_data = {"pos": [2.75, 4.90, 2.75], "color": [1, 1, 1]}
cam_data = {"pos": [278, 273, -800], "up": [0, 1, 0], "dir": [0, 0, 1], "fov": 0.68}
width, height = 1440, 1440

origin, p00, q_w, q_h, light_pos, light_color = build_setup_vectors(light_data, cam_data, width, height)

print(f"Origin: {origin}")
print(f"p00:    {p00}")
print(f"q_w:    {q_w}")
print(f"q_h:    {q_h}")

center_kx, center_ky = width/2, height/2
ray_dir = p00 + q_w * center_kx + q_h * center_ky
ray_dir = ray_dir / np.linalg.norm(ray_dir)
print(f"Center Ray: {ray_dir}")
