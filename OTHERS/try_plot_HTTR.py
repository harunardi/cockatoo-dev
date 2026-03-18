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

# Function to convert 2D hexagonal indexes
def convert_2D_hexx(I_max, J_max, D):
    conv_hexx = [0] * (I_max*J_max)
    tmp_conv = 0
    for j in range(J_max):  
        for i in range(I_max):
            if D[0][j][i] != 0:
                tmp_conv += 1
                m = j * I_max + i
                conv_hexx[m] = tmp_conv

    return conv_hexx

# Function to convert 2D hexagonal indexes to triangular
def convert_2D_tri(I_max, J_max, conv_hexx, level):
    """
    Divide the hexagons into 6 triangles and create a list with numbered index of the variable. 
    This is used to reorder the 2D variable into a column vector.
 
    Parameters
    ----------
    I_max : int
            The size of the column of the list.
    J_max : int
            The size of the row of the list.
    D : list
        The 2D list of diffusion coefficient
 
    Returns
    -------
    conv_tri : list
               The list with numbered index based on the 2D list input.
    D_hexx : list
             The expanded list of diffusion coefficient (triangles)   
    """
    n = 6 * (4 ** (level - 1))

    conv_tri = [0] * I_max * J_max * n
    for j in range(J_max):
        for i in range(I_max):
            m = j * I_max + i
            if conv_hexx[m] != 0:
                for k in range(n):
                    conv_tri[m * n + k] = conv_hexx[m] * n - (n - k - 1)

    conv_hexx_ext = [0] * I_max * J_max * n
    for j in range(J_max):
        for i in range(I_max):
            m = j * I_max + i
            if conv_hexx[m] != 0:
                for k in range(n):
                    conv_hexx_ext[m * n + k] = conv_hexx[m]

    return conv_tri, conv_hexx_ext

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

#######################################################################################################
# Parameters
s = 36.0 # Side-to-side of Hexagon
D1 = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.02541, 0.96422, 0.956443, 0.95258, 0.95751, 0.96562, 1.01815, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.985604, 0.944253, 0.943402, 0.942349, 0.941122, 0.943368, 0.944316, 0.977497, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.968004, 0.941671, 1.39331, 1.11034, 1.28516, 1.10673, 1.374, 0.942043, 0.960087, 0.0], [0.0, 0.0, 0.0, 0.0, 0.945961, 0.941167, 1.10961, 1.31633, 1.2895, 1.28649, 1.30586, 1.10733, 0.941888, 0.950791, 0.0], [0.0, 0.0, 0.0, 0.987457, 0.940077, 1.26897, 1.29044, 1.3029, 1.27157, 1.31619, 1.28747, 1.33394, 0.940767, 0.956993, 0.0], [0.0, 0.0, 0.977238, 0.943275, 1.10596, 1.29078, 1.27406, 1.27123, 1.2741, 1.27191, 1.2912, 1.11153, 0.942495, 0.967259, 0.0], [0.0, 1.08962, 0.942853, 1.28861, 1.30215, 1.31023, 1.26839, 1.30901, 1.26384, 1.30213, 1.31708, 1.39572, 0.94558, 1.06743, 0.0], [0.0, 0.971533, 0.940516, 1.10629, 1.28836, 1.27271, 1.26988, 1.26677, 1.27395, 1.29266, 1.11043, 0.942152, 0.967651, 0.0, 0.0], [0.0, 0.957889, 0.941355, 1.32118, 1.286, 1.30708, 1.27127, 1.31029, 1.29054, 1.26121, 0.93944, 0.966223, 0.0, 0.0, 0.0], [0.0, 0.95136, 0.94164, 1.10903, 1.31249, 1.28557, 1.28848, 1.30384, 1.10792, 0.939835, 0.949434, 0.0, 0.0, 0.0, 0.0], [0.0, 0.954781, 0.943381, 1.39355, 1.10921, 1.28907, 1.10807, 1.29086, 0.941191, 0.96845, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.961098, 0.944596, 0.94325, 0.940643, 0.943988, 0.942479, 0.943061, 0.988751, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.00649, 0.962451, 0.959082, 0.95549, 0.968029, 0.970212, 1.11985, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
D2 = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.800245, 0.797655, 0.797443, 0.797573, 0.797292, 0.798348, 0.800176, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.797967, 0.797538, 0.797718, 0.798278, 0.798564, 0.798238, 0.79804, 0.798178, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.797051, 0.797763, 1.15142, 0.898973, 0.994332, 0.90018, 0.989713, 0.798296, 0.797135, 0.0], [0.0, 0.0, 0.0, 0.0, 0.79725, 0.797954, 0.899085, 0.917898, 0.923933, 0.925359, 0.918562, 0.900524, 0.798353, 0.797711, 0.0], [0.0, 0.0, 0.0, 0.797276, 0.798095, 0.985609, 0.922925, 0.995792, 0.929752, 1.00553, 0.924156, 0.988564, 0.798354, 0.797457, 0.0], [0.0, 0.0, 0.7977, 0.798045, 0.900436, 0.92506, 0.929279, 0.929858, 0.928526, 0.928875, 0.923305, 0.898706, 0.797595, 0.797675, 0.0], [0.0, 0.799961, 0.797869, 0.975536, 0.919457, 0.995622, 0.929686, 0.998382, 0.930921, 0.993958, 0.917346, 1.14957, 0.797328, 0.799999, 0.0], [0.0, 0.798404, 0.798268, 0.900651, 0.924571, 0.929834, 0.929284, 0.930462, 0.929164, 0.9228, 0.899774, 0.797807, 0.79809, 0.0, 0.0], [0.0, 0.797283, 0.798399, 0.986705, 0.924598, 1.00289, 0.929655, 0.99365, 0.924015, 0.9822, 0.798005, 0.79739, 0.0, 0.0, 0.0], [0.0, 0.797287, 0.798287, 0.900163, 0.918485, 0.924261, 0.924592, 0.918213, 0.899635, 0.798381, 0.797338, 0.0, 0.0, 0.0, 0.0], [0.0, 0.797549, 0.797602, 1.15169, 0.899853, 0.990002, 0.900411, 0.976155, 0.797878, 0.797099, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.798096, 0.797417, 0.797503, 0.798475, 0.798022, 0.798193, 0.798176, 0.797784, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.799643, 0.797869, 0.797317, 0.796921, 0.79743, 0.798169, 0.79924, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

s = 36.0 # Side-to-side of Hexagon
h_hexx = s / np.sqrt(3) # Triangle side or hexagon radius
level = 1
h = h_hexx / (2**(level-1))
D = [D1, D2]

I_max = len(D[0][0])
J_max = len(D[0])

n = 6 * (4 ** (level - 1))
N = I_max * J_max
N_hexx = I_max * J_max * n

conv_hexx = convert_2D_hexx(I_max, J_max, D)
conv_tri, conv_hexx_ext = convert_2D_tri(I_max, J_max, conv_hexx, level)
conv_tri_array = np.array(conv_tri)
max_conv = max(conv_tri)

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

## Plot
#plt.figure(figsize=(8, 8))
#plt.triplot(x, y, tri_indices, color='blue', linewidth=0.5)
##plt.scatter(*zip(*hex_centers), color='red', s=10, label='Hex Centers')
#
## Add triangle indices
#for idx, triangle in enumerate(all_triangles):
#    centroid_x = np.mean([v[0] for v in triangle])
#    centroid_y = np.mean([v[1] for v in triangle])
#    plt.text(centroid_x, centroid_y, str(idx), fontsize=4, ha='center', va='center')
#
#plt.axis('equal')
#plt.axis('off')
#plt.savefig("trial_triangulation_new_HTTR.png", dpi=800)
#plt.clf()
plt.figure(figsize=(8, 8))

# Plot triangulation
plt.triplot(x, y, tri_indices, color='blue', linewidth=0.5)

# Add triangle indices
for idx, triangle in enumerate(all_triangles):
    centroid_x = np.mean([v[0] for v in triangle])
    centroid_y = np.mean([v[1] for v in triangle])
    plt.text(centroid_x, centroid_y, str(idx), fontsize=2.5, ha='center', va='center')

# --- Hexagon outlines & labels ---
for idx, (cx, cy) in enumerate(hex_centers):
    # Shift reference hexagon vertices by center
    hx = [vx + cx for vx, vy in hex_vertices]
    hy = [vy + cy for vx, vy in hex_vertices]

    # Bold outline
    plt.plot(hx + [hx[0]], hy + [hy[0]],
             color='black', linewidth=1.0, zorder=2)

    # Hex number in center
    plt.text(cx, cy, str(idx), fontsize=4, fontweight='bold',
             ha='center', va='center', color='red', zorder=3)

plt.axis('equal')
plt.axis('off')
plt.savefig("trial_triangulation_with_hex_numbers.png", dpi=800)
plt.clf()