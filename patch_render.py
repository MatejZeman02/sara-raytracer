import re

with open("src/render_kernel.py", "r") as f:
    text = f.read()

text = text.replace("if False:", "if (x, y) not in DEBUG_PIXELS:\n                    continue\n                print('inspecting', x, y)")

with open("src/render_kernel.py", "w") as f:
    f.write(text)
