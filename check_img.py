import numpy as np
from PIL import Image

img = np.array(Image.open('src/output/output.jpg'))
print("Mean:", img.mean(axis=(0,1)))
print("Std:", img.std(axis=(0,1)))
print("Min:", img.min(axis=(0,1)))
print("Max:", img.max(axis=(0,1)))
