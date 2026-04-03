import re

# Fix Intersection cracks
with open("src/intersection.py", "r") as f:
    text = f.read()

text = text.replace("if u < ZERO or u > ONE:", "if u < -DET_EPSILON or u > ONE + DET_EPSILON:")
text = text.replace("if v < ZERO or u + v > ONE:", "if v < -DET_EPSILON or u + v > ONE + DET_EPSILON:")

with open("src/intersection.py", "w") as f:
    f.write(text)

# Fix Ray generation
with open("src/setup_vectors.py", "r") as f:
    text2 = f.read()

text2 = text2.replace("b_vec = np.cross(dir_vec, up_vec)", "r_vec = np.cross(up_vec, dir_vec); r_vec = r_vec / np.linalg.norm(r_vec)")
text2 = text2.replace("b_vec = b_vec / np.linalg.norm(b_vec)", "")
text2 = text2.replace("q_w = (g_w / (width - 1)) * b_vec", "q_w = (g_w / (width - 1)) * r_vec")
text2 = text2.replace("q_h = (g_h / (height - 1)) * up_vec", "q_h = -(g_h / (height - 1)) * up_vec")
text2 = text2.replace("- (g_w / float32(2.0)) * b_vec", "- (g_w / float32(2.0)) * r_vec")
text2 = text2.replace("-(g_w / float32(2.0)) * b_vec", "- (g_w / float32(2.0)) * r_vec")

with open("src/setup_vectors.py", "w") as f:
    f.write(text2)
