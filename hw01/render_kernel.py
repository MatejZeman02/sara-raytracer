"""CUDA render kernel for hw01."""

from numba import cuda  # type: ignore

from utils.vec_utils import vec3, add, cross, dot, mul, normalize, sub, linear_to_srgb
from intersection import intersect_triangle
from shading import cook_torrance_shading, phong_shading


@cuda.jit
def render_kernel(
    triangles,
    mat_indices,
    materials,
    light_pos,
    light_color,
    p00,
    qw,
    qh,
    origin,
    fb,
    width,
    height,
):
    """Render kernel that casts rays and shades pixels (ray tracing)."""
    # Out of bounds check:
    x, y = cuda.grid(2)
    if x >= width or y >= height:
        return

    # Ray direction from camera space
    dir_x = p00[0] + x * qw[0] - y * qh[0]
    dir_y = p00[1] + x * qw[1] - y * qh[1]
    dir_z = p00[2] + x * qw[2] - y * qh[2]

    ray_dir = normalize(vec3(dir_x, dir_y, dir_z))
    ray_origin = vec3(origin[0], origin[1], origin[2])

    # Find the closest triangle intersection
    closest_t = 1e20
    hit_idx = -1

    for i in range(triangles.shape[0]):
        a = vec3(triangles[i, 0, 0], triangles[i, 0, 1], triangles[i, 0, 2])
        b = vec3(triangles[i, 1, 0], triangles[i, 1, 1], triangles[i, 1, 2])
        c = vec3(triangles[i, 2, 0], triangles[i, 2, 1], triangles[i, 2, 2])

        t, _u, _v = intersect_triangle(ray_origin, ray_dir, a, b, c)
        if 0.001 < t < closest_t:
            closest_t = t
            hit_idx = i

    # No hit: render background dark gray (instead of black)
    if hit_idx == -1:
        fb[y, x, 0] = 20
        fb[y, x, 1] = 20
        fb[y, x, 2] = 20
        return
    # Hit: shade the pixel
    mat_idx = mat_indices[hit_idx]

    ke_r = materials[mat_idx, 7]
    ke_g = materials[mat_idx, 8]
    ke_b = materials[mat_idx, 9]

    # Emmissive?
    if ke_r > 0.0 or ke_g > 0.0 or ke_b > 0.0:
        # They said: just render emissive color
        fb[y, x, 0] = min(255, int(ke_r * 255))
        fb[y, x, 1] = min(255, int(ke_g * 255))
        fb[y, x, 2] = min(255, int(ke_b * 255))
    else:
        # Phong shading for non-emissive
        a = vec3(
            triangles[hit_idx, 0, 0],
            triangles[hit_idx, 0, 1],
            triangles[hit_idx, 0, 2],
        )
        b = vec3(
            triangles[hit_idx, 1, 0],
            triangles[hit_idx, 1, 1],
            triangles[hit_idx, 1, 2],
        )
        c = vec3(
            triangles[hit_idx, 2, 0],
            triangles[hit_idx, 2, 1],
            triangles[hit_idx, 2, 2],
        )

        # Compute surface normal
        n = normalize(cross(sub(b, a), sub(c, a)))
        # CHECK: normal faces towards the ray origin (not needed)
        # if dot(ray_dir, n) > 0.0:
        #     n = mul(n, -1.0)

        # Hit point
        p = add(ray_origin, mul(ray_dir, closest_t))

        # Light direction and view vector
        l_pos = vec3(light_pos[0], light_pos[1], light_pos[2])
        d_l = normalize(sub(l_pos, p))
        v_vec = normalize(sub(ray_origin, p))

        # Material properties:
        r_d = vec3(materials[mat_idx, 0], materials[mat_idx, 1], materials[mat_idx, 2])
        r_s = vec3(materials[mat_idx, 3], materials[mat_idx, 4], materials[mat_idx, 5])
        h_val = materials[mat_idx, 6]
        l_color = vec3(light_color[0], light_color[1], light_color[2])

        # Shading:
        # color = phong_shading(n, v_vec, d_l, r_d, r_s, h_val, l_color)
        color = cook_torrance_shading(n, v_vec, d_l, r_d, r_s, h_val, l_color)

        # apply transfer function before quantizing
        r_srgb = linear_to_srgb(color[0])
        g_srgb = linear_to_srgb(color[1])
        b_srgb = linear_to_srgb(color[2])

        fb[y, x, 0] = min(255, int(r_srgb * 255))
        fb[y, x, 1] = min(255, int(g_srgb * 255))
        fb[y, x, 2] = min(255, int(b_srgb * 255))
