import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from shapely.geometry import Polygon, Point

# Prevent .pyc file generation
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True

#######################################################################################################
def generate_pointy_hex_grid(flat_to_flat_distance, I_max, J_max):
    radius = flat_to_flat_distance / np.sqrt(3)

    hex_vertices = [
        (radius * np.cos(np.pi/6 + 2 * np.pi * k / 6),
         radius * np.sin(np.pi/6 + 2 * np.pi * k / 6))
        for k in range(6)
    ]

    hex_centers = []
    for j in range(J_max):
        for i in range(I_max):
            x_offset = radius * np.sqrt(3) * i + radius * np.sqrt(3) / 2 * j
            y_offset = radius * 1.5 * j
            hex_centers.append((x_offset, y_offset))

    return hex_centers, hex_vertices

def subdivide_triangle(p1, p2, p3, level):
    if level == 1:
        return [(p1, p2, p3)]
    
    mid1 = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
    mid2 = ((p2[0] + p3[0]) / 2, (p2[1] + p3[1]) / 2)
    mid3 = ((p3[0] + p1[0]) / 2, (p3[1] + p1[1]) / 2)
    
    return (subdivide_triangle(p1, mid1, mid3, level - 1) +
            subdivide_triangle(mid1, p2, mid2, level - 1) +
            subdivide_triangle(mid3, mid2, p3, level - 1) +
            subdivide_triangle(mid1, mid2, mid3, level - 1))

def subdivide_pointy_hexagon(center, vertices, level):
    triangles = []
    for i in range(len(vertices)):
        p1 = center
        p2 = vertices[i]
        p3 = vertices[(i + 1) % len(vertices)]
        triangles += subdivide_triangle(p1, p2, p3, level)
    return triangles

def round_vertex(vertex, precision=6):
    return tuple(round(coord, precision) for coord in vertex)

def find_triangle_neighbors_global(triangles, precision=6):
    edge_map = {}
    neighbors = {i: [-1, -1, -1] for i in range(len(triangles))}

    for tri_idx, vertices in enumerate(triangles):
        vertices = [round_vertex(v, precision) for v in vertices]
        edges = [
            tuple(sorted((vertices[0], vertices[1]))),
            tuple(sorted((vertices[1], vertices[2]))),
            tuple(sorted((vertices[2], vertices[0]))),
        ]
        for edge in edges:
            if edge in edge_map:
                neighbor_idx = edge_map[edge]
                for i in range(3):
                    if neighbors[tri_idx][i] == -1:
                        neighbors[tri_idx][i] = neighbor_idx
                        break
                for i in range(3):
                    if neighbors[neighbor_idx][i] == -1:
                        neighbors[neighbor_idx][i] = tri_idx
                        break
            else:
                edge_map[edge] = tri_idx
    return neighbors

def find_triangle_ownership(triangles, hex_centers):
    triangle_ownership = {}
    for idx, triangle in enumerate(triangles):
        centroid = np.mean(triangle, axis=0)
        distances = [np.linalg.norm(np.array(centroid) - np.array(center)) for center in hex_centers]
        hex_index = np.argmin(distances)
        triangle_ownership[idx] = hex_index
    return triangle_ownership


#######################################################################################################
# Parameters
s = 50
I_max, J_max = 2, 2
level = 2
conv_hexx = [1,1,1,1]

#######################################################################################################
# Generate grid
hex_centers, hex_vertices = generate_pointy_hex_grid(s, I_max, J_max)

# Subdivide hexagons
all_triangles = []
for i, center in enumerate(hex_centers):
    if conv_hexx[i] != 0:
        shifted_vertices = [(vx + center[0], vy + center[1]) for vx, vy in hex_vertices]
        all_triangles += subdivide_pointy_hexagon(center, shifted_vertices, level)

triangle_neighbors_global = find_triangle_neighbors_global(all_triangles, precision=6)
triangle_ownership = find_triangle_ownership(all_triangles, hex_centers)

# Extract triangle coordinates
x = [v[0] for triangle in all_triangles for v in triangle]
y = [v[1] for triangle in all_triangles for v in triangle]
tri_indices = np.arange(len(x)).reshape(-1, 3)

#######################################################################################################
# Plot
plt.figure(figsize=(8, 8))
plt.triplot(x, y, tri_indices, color='blue', linewidth=0.4)

# Draw clean outer hexagon boundaries
for i, center in enumerate(hex_centers):
    if conv_hexx[i] == 0:
        continue
    hex_x = [vx + center[0] for vx, vy in hex_vertices] + [hex_vertices[0][0] + center[0]]
    hex_y = [vy + center[1] for vx, vy in hex_vertices] + [hex_vertices[0][1] + center[1]]
    plt.plot(hex_x, hex_y, color='red', linewidth=2)

# Optional triangle labels
for idx, triangle in enumerate(all_triangles):
    cx = np.mean([v[0] for v in triangle])
    cy = np.mean([v[1] for v in triangle])
    plt.text(cx, cy, str(idx), fontsize=7, ha='center', va='center')

plt.axis('equal')
plt.axis('off')
plt.savefig("trial_triangulation_new_hexborder.png", dpi=800)
plt.clf()
