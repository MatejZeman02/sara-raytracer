import re
with open("src/render_kernel.py", "r") as f: text = f.read()

target = """            closest_t, hit_idx, hit_u, hit_v, tri_tests, node_tests = get_closest_hit(
                triangles,
                bvh_nodes,
                use_bvh,
                ray_origin,
                ray_dir,
                inv_rd,
                stack,
                is_primary,
            )"""

replacement = target + """
            if (x, y) in [(47, 457), (47, 461)] and bounce == 0:
                print("hit_idx for", x, y, "is", hit_idx)
"""
if target in text:
    print("Found target!")
    text = text.replace(target, replacement)
    with open("src/render_kernel.py", "w") as f: f.write(text)
else:
    print("Not found!")
