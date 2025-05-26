def generate_octahedron_obj(filename="octahedron.obj"):
    # 6 Vertices des Oktaeders
    vertices = [
        ( 1.0,  0.0,  0.0),
        (-1.0,  0.0,  0.0),
        ( 0.0,  1.0,  0.0),
        ( 0.0, -1.0,  0.0),
        ( 0.0,  0.0,  1.0),
        ( 0.0,  0.0, -1.0),
    ]
    # 8 Faces, Indizes reversed für clockwise Winding
    faces = [
        (1, 5, 3),
        (3, 5, 2),
        (2, 5, 4),
        (4, 5, 1),
        (3, 6, 1),
        (6, 3, 2),
        (6, 2, 4),
        (6, 4, 1),
    ]
    
    with open(filename, 'w') as f:
        f.write("# Octahedron mesh (clockwise faces)\n")
        for x, y, z in vertices:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for i, j, k in faces:
            f.write(f"f {i} {j} {k}\n")
    print(f"Created '{filename}' with {len(vertices)} vertices and {len(faces)} faces.")

if __name__ == "__main__":
    generate_octahedron_obj()
