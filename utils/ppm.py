"""ppm save support"""

import numpy as np


def save_ppm(filename, image_array):
    """Save a 3D numpy array as a PPM image file."""
    image_array = np.clip(image_array, 0, 255).astype(np.uint8)
    height, width, _ = image_array.shape

    with open(filename, "wb") as f:
        # P6 (magic number) for binary PPM - faster than P3
        f.write(f"P6\n{width} {height}\n255\n".encode())
        f.write(image_array.tobytes())
