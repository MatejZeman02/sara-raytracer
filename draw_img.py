from PIL import Image
import numpy as np

img = Image.open('src/output/output.jpg').convert('L')
img = img.resize((40, 20))
arr = np.array(img)
chars = ' .:-=+*#%@'
for row in arr:
    print(''.join([chars[int(p/256*10)] for p in row]))
