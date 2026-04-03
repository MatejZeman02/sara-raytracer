import numpy as np

def intersect_aabb_old(ro, inv_rd, bmin, bmax):
    t1 = (bmin - ro) * inv_rd
    t2 = (bmax - ro) * inv_rd
    tmin_x = min(t1[0], t2[0]); tmax_x = max(t1[0], t2[0])
    tmin_y = min(t1[1], t2[1]); tmax_y = max(t1[1], t2[1])
    tmin_z = min(t1[2], t2[2]); tmax_z = max(t1[2], t2[2])
    tmin = max(tmin_x, max(tmin_y, tmin_z))
    tmax = min(tmax_x, min(tmax_y, tmax_z))
    hit = tmax >= max(tmin, 0.0)
    return hit, tmin

def intersect_aabb_new(ro, inv_rd, bmin, bmax):
    tx1 = (bmin[0] - ro[0]) * inv_rd[0]; tx2 = (bmax[0] - ro[0]) * inv_rd[0]
    tmin_x = tx1 if tx1 < tx2 else tx2; tmax_x = tx1 if tx1 > tx2 else tx2
    ty1 = (bmin[1] - ro[1]) * inv_rd[1]; ty2 = (bmax[1] - ro[1]) * inv_rd[1]
    tmin_y = ty1 if ty1 < ty2 else ty2; tmax_y = ty1 if ty1 > ty2 else ty2
    tz1 = (bmin[2] - ro[2]) * inv_rd[2]; tz2 = (bmax[2] - ro[2]) * inv_rd[2]
    tmin_z = tz1 if tz1 < tz2 else tz2; tmax_z = tz1 if tz1 > tz2 else tz2
    
    m1 = tmin_y if tmin_y > tmin_z else tmin_z
    tmin = tmin_x if tmin_x > m1 else m1
    
    m2 = tmax_y if tmax_y < tmax_z else tmax_z
    tmax = tmax_x if tmax_x < m2 else m2
    
    tmin0 = tmin if tmin > 0.0 else 0.0
    return tmax >= tmin0, tmin

np.random.seed(0)
inv_rd = np.array([np.nan, 1.0, 1.0])
ro = np.array([-1, 0, 0], dtype=np.float32)
bmin = np.array([-1, -1, -1], dtype=np.float32)
bmax = np.array([1, 1, 1], dtype=np.float32)

print("Old:", intersect_aabb_old(ro, inv_rd, bmin, bmax))
print("New:", intersect_aabb_new(ro, inv_rd, bmin, bmax))

