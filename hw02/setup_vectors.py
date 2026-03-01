"""Camera preparation helpers."""

import numpy as np


def build_setup_vectors(light_data, cam_data, width, height):
    fov = cam_data["fov"]
    origin = np.array(cam_data["pos"], dtype=np.float32)
    dir_vec = np.array(cam_data["dir"], dtype=np.float32)
    up_vec = np.array(cam_data["up"], dtype=np.float32)

    b_vec = np.cross(dir_vec, up_vec)
    b_vec = b_vec / np.linalg.norm(b_vec)

    t = 1.0  # focal length
    g_w = 2.0 * t * np.tan(fov / 2.0)
    g_h = g_w * (height / width)

    q_w = (g_w / (width - 1)) * b_vec
    q_h = (g_h / (height - 1)) * up_vec
    p00 = t * dir_vec - (g_w / 2.0) * b_vec + (g_h / 2.0) * up_vec

    light_pos = np.array(light_data["pos"], dtype=np.float32)
    light_color = np.array(light_data["color"], dtype=np.float32)

    return origin, p00, q_w, q_h, light_pos, light_color
