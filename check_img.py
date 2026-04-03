from PIL import Image; import numpy as np; img = np.array(Image.open('src/output/output.jpg')); print('Unique values:', np.unique(img))
