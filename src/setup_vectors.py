"""Camera preparation helpers."""

import numpy as np
from numpy import float32


def build_setup_vectors(light_data, cam_data, width, height):
    fov = float32(cam_data["fov"])
    origin = np.array(cam_data["pos"], dtype=float32)
    dir_vec = np.array(cam_data["dir"], dtype=float32)
    up_vec = np.array(cam_data["up"], dtype=float32)

    b_vec = np.cross(dir_vec, up_vec)
    b_vec = b_vec / np.linalg.norm(b_vec)

    t = float32(1.0)  # focal length
    g_w = float32(2.0) * t * np.tan(fov / float32(2.0))
    g_h = g_w * (height / width)

    q_w = (g_w / (width - 1)) * b_vec
    q_h = (g_h / (height - 1)) * up_vec
    p00 = t * dir_vec - (g_w / float32(2.0)) * b_vec + (g_h / float32(2.0)) * up_vec

    light_pos = np.array(light_data["pos"], dtype=float32)
    light_color = np.array(light_data["color"], dtype=float32)

    return origin, p00, q_w, q_h, light_pos, light_color
