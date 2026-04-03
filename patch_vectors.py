with open("src/setup_vectors.py", "r") as f: text = f.read()
text = text.replace("q_h = (g_h / (height - 1)) * up_vec", "q_h = -(g_h / (height - 1)) * up_vec")
with open("src/setup_vectors.py", "w") as f: f.write(text)
