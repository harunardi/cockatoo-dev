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
    """
    Generate a pointy hexagonal grid using the flat-to-flat distance.
    Parameters:
        flat_to_flat_distance : float
            Flat-to-flat distance of the hexagon.
        I_max, J_max : int
            Number of hexagons along x and y axes.
    Returns:
        hex_centers : list of tuples
            Centers of the hexagons.
        vertices : list of tuples
            Vertices of the hexagon.
    """
    # Calculate radius from flat-to-flat distance
    radius = flat_to_flat_distance / np.sqrt(3)

    # Hexagon vertices (rotated by 30 degrees for pointy-topped)
    hex_vertices = [
        (radius * np.cos(np.pi/6 + 2 * np.pi * k / 6), 
         radius * np.sin(np.pi/6 + 2 * np.pi * k / 6))
        for k in range(6)
    ]

    # Hexagon centers
    hex_centers = []
    for j in range(J_max):
        for i in range(I_max):
            x_offset = radius * np.sqrt(3) * i + radius * np.sqrt(3) / 2 * j
            y_offset = radius * 1.5 * j
            hex_centers.append((x_offset, y_offset))

    return hex_centers, hex_vertices

def subdivide_triangle(p1, p2, p3, level):
    """
    Recursively subdivide a triangle into smaller triangles.
    """
    if level == 1:
        return [(p1, p2, p3)]
    
    mid1 = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
    mid2 = ((p2[0] + p3[0]) / 2, (p2[1] + p3[1]) / 2)
    mid3 = ((p3[0] + p1[0]) / 2, (p3[1] + p1[1]) / 2)
    
    return (
        subdivide_triangle(p1, mid1, mid3, level - 1) +
        subdivide_triangle(mid1, p2, mid2, level - 1) +
        subdivide_triangle(mid3, mid2, p3, level - 1) +
        subdivide_triangle(mid1, mid2, mid3, level - 1)
    )

def subdivide_pointy_hexagon(center, vertices, level):
    """
    Subdivide a pointy hexagon into smaller triangles.
    """
    triangles = []
    for i in range(len(vertices)):
        p1 = center
        p2 = vertices[i]
        p3 = vertices[(i + 1) % len(vertices)]
        triangles += subdivide_triangle(p1, p2, p3, level)
    return triangles

def round_vertex(vertex, precision=6):
    """
    Round vertex coordinates to a fixed precision.
    """
    return tuple(round(coord, precision) for coord in vertex)

def find_triangle_neighbors_global(triangles, precision=6):
    """
    Find neighbors for each triangle globally based on shared edges.
    Assign -1 for neighbors on the boundary.
    Each triangle will have exactly 3 neighbors.
    """
    edge_map = {}
    neighbors = {i: [-1, -1, -1] for i in range(len(triangles))}  # Initialize with -1 for boundaries

    # Step 1: Map edges to triangles
    for tri_idx, vertices in enumerate(triangles):
        vertices = [round_vertex(v, precision) for v in vertices]
        edges = [
            tuple(sorted((vertices[0], vertices[1]))),
            tuple(sorted((vertices[1], vertices[2]))),
            tuple(sorted((vertices[2], vertices[0]))),
        ]
        for edge in edges:
            if edge in edge_map:
                # Shared edge found
                neighbor_idx = edge_map[edge]
                # Assign neighbors for both triangles
                for i in range(3):
                    if neighbors[tri_idx][i] == -1:
                        neighbors[tri_idx][i] = neighbor_idx
                        break
                for i in range(3):
                    if neighbors[neighbor_idx][i] == -1:
                        neighbors[neighbor_idx][i] = tri_idx
                        break
            else:
                # Map the edge to the current triangle
                edge_map[edge] = tri_idx

    return neighbors

def find_triangle_ownership(triangles, hex_centers):
    """
    Assign each triangle to the nearest hexagon center.
    """
    triangle_ownership = {}
    for idx, triangle in enumerate(triangles):
        centroid = np.mean(triangle, axis=0)
        distances = [np.linalg.norm(np.array(centroid) - np.array(center)) for center in hex_centers]
        hex_index = np.argmin(distances)
        triangle_ownership[idx] = hex_index
    return triangle_ownership

def find_boundary_for_each_hex(triangle_neighbors, ownership):
    """
    Identify boundary triangles for each hexagon by checking neighboring triangle ownership.
    """
    hex_boundaries = {i: [] for i in set(ownership.values())}
    hex_vert_boundaries = {i: [] for i in set(ownership.values())}
    
    for tri_idx, neighbors in triangle_neighbors.items():
        hex_idx = ownership[tri_idx]
        for neighbor in neighbors:
            if neighbor == -1 or ownership[neighbor] != hex_idx:
                hex_boundaries[hex_idx].append(tri_idx)
                break

    # Step 2: manually insert in-between triangles
    for hex_idx, boundary in hex_boundaries.items():
        new_boundary = []
        for i in range(0, len(boundary), 2):
            pair = boundary[i:i+2]
            new_boundary.extend(pair)
            if len(pair) == 2:
                new_boundary.append(max(pair) + 2)
        hex_boundaries[hex_idx] = new_boundary
        
    # Loop through the boundary triangles and classify vertical boundaries
    for hex_idx, boundary in hex_boundaries.items():
        for i, tri_idx in enumerate(boundary):
            if (i // 2**(level-1) == 2) or (i // 2**(level-1) == 5):
                hex_vert_boundaries[hex_idx].append(tri_idx)

    return hex_boundaries, hex_vert_boundaries

#######################################################################################################
# Parameters
s = 50
I_max, J_max = 2, 2
level = 2
#conv_hexx = [1,1,1,1]
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

# Find neighbors with debugging
triangle_neighbors_global = find_triangle_neighbors_global(all_triangles, precision=6)

# Display neighbors
print("\nTriangle Neighbors (Global):")
conv_neighbor_new = []
for idx, neighbors in triangle_neighbors_global.items():
#    print(f"Triangle {idx}: Neighbors -> {neighbors}")
    conv_neighbor_new.append(neighbors)

# Extract triangle coordinates for plotting
x = [v[0] for triangle in all_triangles for v in triangle]
y = [v[1] for triangle in all_triangles for v in triangle]
tri_indices = np.arange(len(x)).reshape(-1, 3)

# Plot
plt.figure(figsize=(8, 8))
plt.triplot(x, y, tri_indices, color='blue', linewidth=0.5)
#plt.scatter(*zip(*hex_centers), color='red', s=10, label='Hex Centers')

# Add triangle indices
for idx, triangle in enumerate(all_triangles):
    centroid_x = np.mean([v[0] for v in triangle])
    centroid_y = np.mean([v[1] for v in triangle])
    plt.text(centroid_x, centroid_y, str(idx), fontsize=8, ha='center', va='center')

plt.axis('equal')
plt.axis('off')
plt.savefig("trial_triangulation_new.png", dpi=800)
plt.clf()

triangle_ownership = find_triangle_ownership(all_triangles, hex_centers)
hex_boundary_triangles, hex_vert_boundary_triangles = find_boundary_for_each_hex(triangle_neighbors_global, triangle_ownership)

for hex_idx, boundary in hex_boundary_triangles.items():
    print(f"Hexagon {hex_idx}: Boundary Triangles -> {boundary}")

for hex_idx, boundary in hex_vert_boundary_triangles.items():
    print(f"Hexagon {hex_idx}: Vertical Boundary Triangles -> {boundary}")

#def triangles_within_boundary(triangles, hex_vertices, threshold=1.0):
#    """
#    Find triangles within `threshold` distance of the hexagon boundary.
#    
#    Parameters:
#        triangles : list of tuples
#            Each triangle is ((x1,y1),(x2,y2),(x3,y3))
#        hex_vertices : list of tuples
#            Vertices of the hexagon
#        threshold : float
#            Distance in same units as coordinates (default = 1.0)
#    
#    Returns:
#        boundary_tris : list of triangles within threshold distance of boundary
#    """
#    hex_poly = Polygon(hex_vertices)
#    shrunk_hex = hex_poly.buffer(-threshold)
#
#    boundary_tris = []
#    for tri in triangles:
#        tri_poly = Polygon(tri)
#        # Check if inside hexagon but not inside shrunk region
#        if hex_poly.contains(tri_poly) and not shrunk_hex.contains(tri_poly):
#            boundary_tris.append(tri)
#    return boundary_tris

def triangle_indices_within_boundary(triangles, hex_vertices, threshold=1.0):
    """
    Get local indices of triangles within `threshold` distance 
    of the hexagon boundary.
    
    Parameters:
        triangles : list of tuples
            Each triangle is ((x1,y1),(x2,y2),(x3,y3))
        hex_vertices : list of tuples
            Vertices of the hexagon
        threshold : float
            Distance in same units as coordinates (default = 1.0)
    
    Returns:
        boundary_indices : list of int
            Local indices of boundary triangles
    """
    hex_poly = Polygon(hex_vertices)
    shrunk_hex = hex_poly.buffer(-threshold)

    boundary_indices = []
    for idx, tri in enumerate(triangles):  # local index
        tri_poly = Polygon(tri)
        if hex_poly.contains(tri_poly) and not shrunk_hex.contains(tri_poly):
            boundary_indices.append(idx)
    return boundary_indices

# Example usage
# Get triangles near the boundary (1 mm inside)
boundary_tris = triangle_indices_within_boundary(all_triangles, 
                                          [(vx+center[0], vy+center[1]) for vx,vy in hex_vertices],
                                          threshold=1.0)


print("Total triangles:", len(all_triangles))
print("Boundary triangles:", len(boundary_tris))
print("Boundary triangles indices:")
print(boundary_tris)


def triangle_indices_within_boundary_fixed(triangles, hex_vertices, threshold=1.0):
    """
    Get indices of triangles whose any vertex is within `threshold`
    distance from the hexagon boundary.
    """
    hex_poly = Polygon(hex_vertices)
    boundary_indices = []

    for idx, tri in enumerate(triangles):
        # Check distance of each vertex to hexagon boundary
        for vertex in tri:
            point = Point(vertex)
            distance = hex_poly.exterior.distance(point)
            if distance <= threshold:
                boundary_indices.append(idx)
                break  # No need to check other vertices
    
    return boundary_indices

def reference_hex_boundary_triangles(flat_to_flat_distance, level, threshold=1.0):
    """
    Generate triangles for a single reference hexagon at (0,0) 
    and return indices of triangles near the boundary.
    
    Parameters:
        flat_to_flat_distance : float
            Flat-to-flat distance of the hexagon
        level : int
            Subdivision level for triangles
        threshold : float
            Distance threshold for boundary detection
    
    Returns:
        triangles : list of tuples
            All triangles of the hexagon
        boundary_indices : list of int
            Indices of triangles near the boundary
    """
    # Step 1: Create reference hexagon at (0,0)
    radius = flat_to_flat_distance / np.sqrt(3)
    hex_vertices = [
        (radius * np.cos(np.pi/6 + 2 * np.pi * k / 6), 
         radius * np.sin(np.pi/6 + 2 * np.pi * k / 6))
        for k in range(6)
    ]
    center = (0, 0)

    # Step 2: Subdivide hexagon into triangles
    triangles = subdivide_pointy_hexagon(center, hex_vertices, level)

    # Step 3: Find boundary triangles
    boundary_indices = triangle_indices_within_boundary_fixed(triangles, hex_vertices, threshold)

    return triangles, boundary_indices

print("hex_vertices:", hex_vertices[0])
boundary_tris_new = reference_hex_boundary_triangles(s, level, threshold=1.0)
print("Reference hexagon boundary triangles indices:")
print(boundary_tris_new[1])  # Print only the indices