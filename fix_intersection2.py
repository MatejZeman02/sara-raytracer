with open("src/intersection.py", "r") as f:
    text = f.read()

import re

new_code = """
@device_jit
def fmin(a, b):
    # returns b if a is NaN, else if b is NaN returns a, else min(a,b)
    return b if a != a else (a if b != b else (a if a < b else b))

@device_jit
def fmax(a, b):
    return b if a != a else (a if b != b else (a if a > b else b))

@device_jit
def intersect_aabb(ro, inv_rd, bmin, bmax):
    \"\"\"Ray-AABB intersection test using the slab method.\"\"\"
    t1_x = (bmin[0] - ro[0]) * inv_rd[0]
    t2_x = (bmax[0] - ro[0]) * inv_rd[0]
    tmin_x = fmin(t1_x, t2_x)
    tmax_x = fmax(t1_x, t2_x)

    t1_y = (bmin[1] - ro[1]) * inv_rd[1]
    t2_y = (bmax[1] - ro[1]) * inv_rd[1]
    tmin_y = fmin(t1_y, t2_y)
    tmax_y = fmax(t1_y, t2_y)

    t1_z = (bmin[2] - ro[2]) * inv_rd[2]
    t2_z = (bmax[2] - ro[2]) * inv_rd[2]
    tmin_z = fmin(t1_z, t2_z)
    tmax_z = fmax(t1_z, t2_z)

    # slab method overlap check
    tmin = fmax(tmin_x, fmax(tmin_y, tmin_z))
    tmax = fmin(tmax_x, fmin(tmax_y, tmax_z))

    hit = tmax >= fmax(tmin, ZERO)

    return hit, tmin"""

text = re.sub(r'@device_jit\ndef intersect_aabb.*?return hit, tmin', new_code, text, flags=re.DOTALL)

with open("src/intersection.py", "w") as f:
    f.write(text)
