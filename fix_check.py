import numpy as np
from PIL import Image

def get_stats(f):
    img = np.array(Image.open(f))
    return img.mean(axis=(0,1)), img.std(axis=(0,1))

print("If there are weird pixels, let's see min/max arrays or NaNs")
