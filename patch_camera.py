import re
with open("src/setup_vectors.py", "r") as f: text = f.read()

text = text.replace("b_vec = np.cross(dir_vec, up_vec)", "r_vec = np.cross(up_vec, dir_vec); r_vec = r_vec / np.linalg.norm(r_vec)")
text = text.replace("b_vec = b_vec / np.linalg.norm(b_vec)", "")
text = text.replace("q_w = (g_w / (width - 1)) * b_vec", "q_w = (g_w / (width - 1)) * r_vec")
text = text.replace("q_h = (g_h / (height - 1)) * up_vec", "q_h = -(g_h / (height - 1)) * up_vec")
text = text.replace("-(g_w / float32(2.0)) * b_vec", "- (g_w / float32(2.0)) * r_vec")

with open("src/setup_vectors.py", "w") as f: f.write(text)
