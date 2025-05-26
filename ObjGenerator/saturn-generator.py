import numpy as np

def generate_uv_sphere(radius=1.0, lat_segments=10, lon_segments=20):
    vertices = []
    faces = []
    for i in range(lat_segments + 1):
        theta = np.pi * i / lat_segments
        sin_theta, cos_theta = np.sin(theta), np.cos(theta)
        for j in range(lon_segments):
            phi = 2 * np.pi * j / lon_segments
            sin_phi, cos_phi = np.sin(phi), np.cos(phi)
            x = radius * sin_theta * cos_phi
            y = radius * cos_theta
            z = radius * sin_theta * sin_phi
            vertices.append((x, y, z))
    for i in range(lat_segments):
        for j in range(lon_segments):
            p1 = i * lon_segments + j
            p2 = p1 + lon_segments
            p3 = p2 + 1 if (j + 1) < lon_segments else p2 + 1 - lon_segments
            p4 = p1 + 1 if (j + 1) < lon_segments else p1 + 1 - lon_segments
            if i != 0:
                faces.append((p1 + 1, p2 + 1, p4 + 1))
            if i != lat_segments - 1:
                faces.append((p4 + 1, p2 + 1, p3 + 1))
    return vertices, faces

def generate_ring(inner_radius=1.2, outer_radius=1.5, segments=50):
    vertices = []
    faces = []
    for r in (inner_radius, outer_radius):
        for i in range(segments):
            angle = 2 * np.pi * i / segments
            x = r * np.cos(angle)
            y = 0.0
            z = r * np.sin(angle)
            vertices.append((x, y, z))
    base = 0
    for i in range(segments):
        i_next = (i + 1) % segments
        v0 = base + i
        v1 = base + i_next
        v2 = base + segments + i_next
        v3 = base + segments + i
        faces.append((v0 + 1, v1 + 1, v2 + 1))
        faces.append((v0 + 1, v2 + 1, v3 + 1))
    return vertices, faces

# Erzeuge Planet und Ring
planet_verts, planet_faces = generate_uv_sphere(lat_segments=10, lon_segments=20)
ring_verts, ring_faces = generate_ring(segments=50)

# Kombiniere
all_verts = planet_verts + ring_verts
all_faces = planet_faces + [(f[0] + len(planet_verts), f[1] + len(planet_verts), f[2] + len(planet_verts)) for f in ring_faces]

# Schreibe OBJ
with open('planet_with_ring.obj', 'w') as f:
    for v in all_verts:
        f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
    for face in all_faces:
        f.write(f"f {face[0]} {face[1]} {face[2]}\n")

print(f"Erzeugt OBJ mit {len(all_verts)} Vertices und {len(all_faces)} Faces: planet_with_ring.obj")
