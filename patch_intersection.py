with open("src/intersection.py", "r") as f: text = f.read()
text = text.replace("if u < ZERO or u > ONE:", "if u < -DET_EPSILON or u > ONE + DET_EPSILON:")
text = text.replace("if v < ZERO or u + v > ONE:", "if v < -DET_EPSILON or u + v > ONE + DET_EPSILON:")
with open("src/intersection.py", "w") as f: f.write(text)
