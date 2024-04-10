import numpy as np


def random(radius, num_points):
    points = []
    for _ in range(num_points):
        theta = np.random.uniform(0, np.pi)
        phi = np.random.uniform(0, 2 * np.pi)

        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta)

        points.append((x, y, z))

    return np.array(points)

def uniform(radius, delta_phi, delta_theta):
    """
    Generate points on a sphere in Cartesian coordinates.

    :param delta_phi: Step size for azimuthal angle (phi) in radians.
    :param delta_theta: Step size for polar angle (theta) in radians.
    :param radius: Radius of the sphere.
    :return: NumPy array of shape (N, 3) where N is the number of points.
    """
    phi_values = np.arange(0, 2 * np.pi, delta_phi)
    theta_values = np.arange(0, np.pi + delta_theta, delta_theta)  # include pi for the last point

    # Create meshgrid
    phi, theta = np.meshgrid(phi_values, theta_values)

    # Cartesian coordinates
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    # Flatten and stack to get a (N, 3) shape
    points = np.column_stack((x.flatten(), y.flatten(), z.flatten()))

    return points

def subdivide_triangles(vertices, faces):
    new_vertices = []
    new_faces = []
    vertex_map = {}

    for face in faces:
        face_vertices = [tuple(vertices[idx]) for idx in face]

        # Get or create new vertex indices
        midpoints = []
        for i in range(3):
            edge = (face_vertices[i], face_vertices[(i + 1) % 3])
            edge = tuple(sorted(edge))

            if edge not in vertex_map:
                new_vertex = (np.array(edge[0]) + np.array(edge[1])) / 2
                new_vertex /= np.linalg.norm(new_vertex)
                new_vertices.append(new_vertex)
                vertex_map[edge] = len(vertices) + len(new_vertices) - 1

            midpoints.append(vertex_map[edge])

        # Create new faces
        a, b, c = midpoints
        new_faces.extend([
            [face[0], a, c],
            [a, face[1], b],
            [c, b, face[2]],
            [a, b, c]
        ])

    vertices = np.vstack([vertices, new_vertices])
    return vertices, np.array(new_faces)

def icosahedral(radius, subdivisions):
    t = (1.0 + np.sqrt(5.0)) / 2.0

    # Create vertices of an icosahedron
    vertices = np.array([[-1,  t,  0],
                         [ 1,  t,  0],
                         [-1, -t,  0],
                         [ 1, -t,  0],
                         [ 0, -1,  t],
                         [ 0,  1,  t],
                         [ 0, -1, -t],
                         [ 0,  1, -t],
                         [ t,  0, -1],
                         [ t,  0,  1],
                         [-t,  0, -1],
                         [-t,  0,  1]])

    # Normalize vertices to lie on the sphere surface
    vertices /= np.linalg.norm(vertices, axis=1)[:, np.newaxis]

    # Define the faces of the icosahedron
    faces = np.array([[0, 11, 5],
                      [0, 5, 1],
                      [0, 1, 7],
                      [0, 7, 10],
                      [0, 10, 11],
                      [1, 5, 9],
                      [5, 11, 4],
                      [11, 10, 2],
                      [10, 7, 6],
                      [7, 1, 8],
                      [3, 9, 4],
                      [3, 4, 2],
                      [3, 2, 6],
                      [3, 6, 8],
                      [3, 8, 9],
                      [4, 9, 5],
                      [2, 4, 11],
                      [6, 2, 10],
                      [8, 6, 7],
                      [9, 8, 1]])

    # Subdivide each triangle in the icosahedron
    for _ in range(subdivisions):
        vertices, faces = subdivide_triangles(vertices, faces)

    # Scale points to the desired radius
    sphere_points = vertices * radius
    return np.array(sphere_points)

def fibonacci(radius, num_points):
    points = []
    phi = np.pi * (3 - np.sqrt(5))  # golden angle in radians

    for i in range(num_points):
        y = 1 - (i / float(num_points - 1)) * 2  # y goes from 1 to -1
        radius_at_y = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius_at_y
        z = np.sin(theta) * radius_at_y

        points.append((radius * x, radius * y, radius * z))

    return np.array(points)