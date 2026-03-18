import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csc_matrix
import os
import sys
import h5py
from scipy.interpolate import RBFInterpolator
from scipy.interpolate import griddata
from shapely.geometry import Polygon, Point
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

# Prevent .pyc file generation
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True

##############################################################################
# Function to save sparse matrix to file
def save_sparse_matrix(A, filename):
    """
    Save a sparse matrix into a file.
 
    Parameters
    ----------
    A : sparse matrix
        The matrix to be saved.
    b : string
        The file name of the sparse matrix.
 
    Returns
    -------
    file (txt)
        The file containing the sparse matrix.    
    """

    A_coo = A.tocoo()
    I, J, V = A_coo.row, A_coo.col, A_coo.data
    with open(filename, 'w') as file:
        for i, j, v in zip(I, J, V):
            file.write(f"{i+1} {j+1} {v}\n")
    
    print(f"Sparse matrix saved to {filename}")

# Function to convert 3D hexagonal indexes
def convert_3D_hexx(K_max, J_max, I_max, D):
    conv_hexx = [0] * (K_max * J_max * I_max)
    tmp_conv = 0
    for k in range(K_max):
        for j in range(J_max):  
            for i in range(I_max):
                if D[0][k][j][i] != 0:
                    tmp_conv += 1
                    m = k * J_max * I_max + j * I_max + i
                    conv_hexx[m] = tmp_conv

    return conv_hexx

# Function to convert 3D hexagonal indexes to triangular
def convert_3D_tri(K_max, J_max, I_max, conv_hexx, level):
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

    conv_tri = [0] * K_max * J_max * I_max * n
    for k in range(K_max):
        for j in range(J_max):
            for i in range(I_max):
                m = k * J_max * I_max + j * I_max + i
                if conv_hexx[m] != 0:
                    for l in range(n):
                        conv_tri[m * n + l] = conv_hexx[m] * n - (n - l - 1)

    conv_hexx_ext = [0] * K_max * J_max * I_max * n
    for k in range(K_max):
        for j in range(J_max):
            for i in range(I_max):
                m = k * J_max * I_max + j * I_max + i
                if conv_hexx[m] != 0:
                    for l in range(n):
                        conv_hexx_ext[m * n + l] = conv_hexx[m]

    return conv_tri, conv_hexx_ext

# Function to save data in HDF5 format
def save_output_hdf5(filename, output_dict):
    with h5py.File(filename, 'w') as f:
        for key, value in output_dict.items():
            real_data = np.array([complex_number['real'] for complex_number in value])
            imag_data = np.array([complex_number['imaginary'] for complex_number in value])
            f.create_dataset(f'{key}/real', data=real_data)
            f.create_dataset(f'{key}/imaginary', data=imag_data)

# Function to load data in HDF5 format
def load_output_hdf5(filename):
    output_dict = {}
    with h5py.File(filename, 'r') as f:
        for key in f.keys():
            real_data = f[f'{key}/real'][:]
            imag_data = f[f'{key}/imaginary'][:]
            complex_data = [complex(real, imag) for real, imag in zip(real_data, imag_data)]
            output_dict[key] = [{"real": c.real, "imaginary": c.imag} for c in complex_data]
    return output_dict

##############################################################################
def expand_XS_hexx_3D(group, K_max, J_max, I_max, XS, level):
    n = 6 * (4 ** (level - 1))

    XS_temp = np.reshape(XS, (group, K_max, J_max, I_max))
    XS_hexx = [[0] * (K_max * J_max * I_max * n) for _ in range(group)]

    for g in range(group):
        for k in range(K_max):
            for j in range(J_max):
                for i in range(I_max):
                    m = k * (J_max * I_max) + j * I_max + i
                    for l in range(n):
                        XS_hexx[g][m * n + l] = XS_temp[g][k][j][i]

    return XS_hexx

def expand_SIGS_hexx_3D(group, K_max, J_max, I_max, SIGS, level):
    n = 6 * (4 ** (level - 1))

    SIGS_hexx = [[[0] * K_max * I_max * J_max * n for _ in range(group)] for _ in range(group)]
    for gr1 in range(group):
        for gr2 in range(group):
            SIGS_temp = SIGS[gr1][gr2]

            SIGS_temp_hexx = [0] * (K_max * J_max * I_max * n)
            for k in range(K_max):
                for j in range(J_max):
                    for i in range(I_max):
                        m = k * (J_max * I_max) + j * I_max + i
                        for p in range(n):
                            SIGS_temp_hexx[m * n + p] = SIGS_temp[m]

            SIGS_hexx[gr1][gr2] = SIGS_temp_hexx

    return SIGS_hexx
##############################################################################
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

def find_triangle_neighbors_2D(triangles, precision=6):
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

def calculate_neighbors_3D(s, I_max, J_max, K_max, conv_hexx, level):
    """
    Do all the necessary calculations to get triangle neighbors.
    """
    # Generate grid
    hex_centers, hex_vertices = generate_pointy_hex_grid(s, I_max, J_max)

    # Subdivide hexagons
    all_triangles = []
    for i, center in enumerate(hex_centers):
        if conv_hexx[i] != 0:
            shifted_vertices = [(vx + center[0], vy + center[1]) for vx, vy in hex_vertices]
            all_triangles += subdivide_pointy_hexagon(center, shifted_vertices, level)

    # Find neighbors with debugging
    triangle_neighbors_global = find_triangle_neighbors_2D(all_triangles, precision=6)

    # Extract triangle coordinates for plotting
    x = [v[0] for triangle in all_triangles for v in triangle]
    y = [v[1] for triangle in all_triangles for v in triangle]
    tri_indices = np.arange(len(x)).reshape(-1, 3)

    conv_neighbor_2D = []
    conv_tri, conv_hexx_ext = convert_3D_tri(K_max, J_max, I_max, conv_hexx, level)
    for idx, neighbors in triangle_neighbors_global.items():
        conv_neighbor_2D.append(neighbors)

    conv_neighbor_2D_temp = [[[-1] * 3 for _ in range(len(conv_neighbor_2D))] for _ in range(K_max)]
    for k in range(K_max):
        for n in range(len(conv_neighbor_2D)):
            for t in range(3):
                if conv_neighbor_2D[n][t] == -1:
                    pass
                else:
                    conv_neighbor_2D_temp[k][n][t] = conv_neighbor_2D[n][t] + (k * len(conv_neighbor_2D))

    conv_neighbor_3D = [[-1] * 5 for _ in range(max(conv_tri))]
    for n in range(len(conv_neighbor_3D)):
        k = n // (len(conv_neighbor_2D))
        n2 = n % (len(conv_neighbor_2D))
        for t in range(3):
            conv_neighbor_3D[n][t] = conv_neighbor_2D_temp[k][n2][t]
        if 0 < k < K_max - 1:
            conv_neighbor_3D[n][3] = n - (len(conv_neighbor_2D))
            conv_neighbor_3D[n][4] = n + (len(conv_neighbor_2D))
        elif k == 0:
            conv_neighbor_3D[n][4] = n + (len(conv_neighbor_2D))
        elif k == K_max - 1:
            conv_neighbor_3D[n][3] = n - (len(conv_neighbor_2D))

    return conv_neighbor_2D, conv_neighbor_3D, tri_indices, x, y, all_triangles

##############################################################################
def FORWARD_D_3D_hexx_matrix(group, BC, conv_tri, conv_neighbor, h, dz, D, level):
    def DIFCOEFF_TRI_INTERIOR_plane(D_i, D_next, h):
        return (8 / h**2) * ((D_i * D_next) / (D_i + D_next))

    def DIFCOEFF_TRI_BC_plane(D_i, h, BC):
        if BC == 1:  # Zero Flux
            return (8 * D_i) / ((h**2))
        if BC == 2:  # Reflective
            return 0
        if BC == 3:  # Vacuum
            return (8 * D_i) / ((4 * D_i * h * np.sqrt(3)) + (h**2))

    def DIFCOEFF_TRI_INTERIOR_axial(D_i, D_next, dz):
        return (2 / dz**2) * ((D_i * D_next) / (D_i + D_next))

    def DIFCOEFF_TRI_BC_axial(D_i, dz, BC):
        if BC == 1:  # Zero Flux
            return (2 * D_i) / ((dz**2))
        if BC == 2:  # Reflective
            return 0
        if BC == 3:  # Vacuum
            return (2 * D_i) / ((4 * D_i * dz) + (dz**2))

    n = 6 * (4 ** (level - 1))
    I_max = len(D[0][0][0])
    J_max = len(D[0][0])
    K_max = len(D[0])

    D_hexx = [[0] * K_max * J_max * I_max * n for _ in range(group)]
    for g in range(group):
        for k in range(K_max):
            for j in range(J_max):
                for i in range(I_max):
                    m = k * (J_max * I_max) + j * I_max + i
                    for l in range(n):
                        D_hexx[g][m * n + l] = D[g][k][j][i]

    max_conv = max(conv_tri)
    D_hexx_reshaped = [[0] * max_conv for _ in range(group)]
    for g in range(group):
        for m in range(len(conv_tri)):
            if D_hexx[g][m] != 0:
                D_hexx_reshaped[g][conv_tri[m]-1] = D_hexx[g][m]

    # Extract boundary conditions for north, south, east, and west
    BC_north = BC[0]
    BC_south = BC[1]
    BC_east = BC[2]
    BC_west = BC[3]
    BC_top = BC[4]
    BC_bottom = BC[5]

    # Initialize the full matrix with zeros
    matrix = lil_matrix((group * max_conv, group * max_conv))

    # Loop through each group and each point in the conv array
    for g in range(group):
        for i in range(max_conv):
            idx = g * max_conv + i
            D_i = D_hexx_reshaped[g][i]

            # Loop through neighbors
            for neighbor_idx, neighbor in enumerate(conv_neighbor[i]):
                if neighbor_idx < 3:
                    if neighbor < 0:
                        # Handle boundary conditions for points with no neighbors
                        if neighbor == -1:
                            matrix[idx, idx] += DIFCOEFF_TRI_BC_plane(D_i, h, BC_north)
                        elif neighbor == -2:
                            matrix[idx, idx] += DIFCOEFF_TRI_BC_plane(D_i, h, BC_east)
                        elif neighbor == -3:
                            matrix[idx, idx] += DIFCOEFF_TRI_BC_plane(D_i, h, BC_south)
                        elif neighbor == -4:
                            matrix[idx, idx] += DIFCOEFF_TRI_BC_plane(D_i, h, BC_west)
                    else:
                        neighbor_conv_idx = g * max_conv + (neighbor)
                        D_next = D_hexx_reshaped[g][neighbor]
                        diff_coeff = DIFCOEFF_TRI_INTERIOR_plane(D_i, D_next, h)
                        matrix[idx, neighbor_conv_idx] += -diff_coeff
                        matrix[idx, idx] += diff_coeff
                else:
                    if neighbor == -1:
                        matrix[idx, idx] += DIFCOEFF_TRI_BC_axial(D_i, dz, BC_top)
#                        # Handle boundary conditions for points with no neighbors
#                        if neighbor == -5:
#                            matrix[idx, idx] += DIFCOEFF_TRI_BC_axial(D_i, dz, BC_top)
#                        elif neighbor == -6:
#                            matrix[idx, idx] += DIFCOEFF_TRI_BC_axial(D_i, dz, BC_bottom)
                    else:
                        neighbor_conv_idx = g * max_conv + (neighbor)
                        D_next = D_hexx_reshaped[g][neighbor]
                        diff_coeff = DIFCOEFF_TRI_INTERIOR_axial(D_i, D_next, dz)
                        matrix[idx, neighbor_conv_idx] += -diff_coeff
                        matrix[idx, idx] += diff_coeff

    print("D_hexx_mat generated")
    return matrix

def FORWARD_TOT_3D_hexx_matrix(group, K_max, J_max, I_max, conv_tri, TOT, level):
    TOT_hexx = expand_XS_hexx_3D(group, K_max, J_max, I_max, TOT, level)

    max_conv = max(conv_tri)
    matrix = lil_matrix((group * max_conv, group * max_conv))
    for g in range(group):
        for n in range(len(TOT_hexx[0])):
            if TOT_hexx[g][n] != 0:
                matrix[g*max_conv+(conv_tri[n]-1), g*max_conv+(conv_tri[n]-1)] += TOT_hexx[g][n]
    print("TOT_mat generated")
    return matrix

def FORWARD_SCAT_3D_hexx_matrix(group, K_max, J_max, I_max, conv_tri, SIGS_reshaped, level):
    SIGS_hexx = expand_SIGS_hexx_3D(group, K_max, J_max, I_max, SIGS_reshaped, level)

    N = K_max * J_max * I_max
    max_conv = max(conv_tri)
    matrix = lil_matrix((group*max_conv, group*max_conv))
    if group == 1:
        for i in range(N):
            matrix[(conv_tri[k]-1), (conv_tri[k]-1)] += SIGS_hexx[0][i]
    else:
        for i in range(group):
            for j in range(group):
                for k in range(len(conv_tri)):
                    if SIGS_hexx[i][j][k] != 0:
                        matrix[i*max_conv+(conv_tri[k]-1), j*max_conv+(conv_tri[k]-1)] += SIGS_hexx[i][j][k]
    print("SCAT_mat generated")
    return matrix

def FORWARD_NUFIS_3D_hexx_matrix(group, K_max, J_max, I_max, conv_tri, chi, NUFIS, level):
    chi_hexx = expand_XS_hexx_3D(group, K_max, J_max, I_max, chi, level)
    NUFIS_hexx = expand_XS_hexx_3D(group, K_max, J_max, I_max, NUFIS, level)
    
    max_conv = max(conv_tri)
    matrix = lil_matrix((group * max_conv, group * max_conv))
    for i in range(group):
        for j in range(group):
            for k in range(len(NUFIS_hexx[0])):
                if NUFIS_hexx[j][k] != 0:
                    matrix[i*max_conv + (conv_tri[k]-1), j*max_conv + (conv_tri[k]-1)] += chi_hexx[i][k]*NUFIS_hexx[j][k]

    print("F_mat generated")
    return matrix

##############################################################################
def ADJOINT_D_3D_hexx_matrix(group, BC, conv_tri, conv_neighbor, h, dz, D, level):
    def DIFCOEFF_TRI_INTERIOR_plane(D_i, D_next, h):
        return (8 / h**2) * ((D_i * D_next) / (D_i + D_next))

    def DIFCOEFF_TRI_BC_plane(D_i, h, BC):
        if BC == 1:  # Zero Flux
            return (8 * D_i) / ((h**2))
        if BC == 2:  # Reflective
            return 0
        if BC == 3:  # Vacuum
            return (8 * D_i) / ((4 * D_i * h * np.sqrt(3)) + (h**2))

    def DIFCOEFF_TRI_INTERIOR_axial(D_i, D_next, dz):
        return (2 / dz**2) * ((D_i * D_next) / (D_i + D_next))

    def DIFCOEFF_TRI_BC_axial(D_i, dz, BC):
        if BC == 1:  # Zero Flux
            return (2 * D_i) / ((dz**2))
        if BC == 2:  # Reflective
            return 0
        if BC == 3:  # Vacuum
            return (2 * D_i) / ((4 * D_i * dz) + (dz**2))

    n = 6 * (4 ** (level - 1))
    I_max = len(D[0][0][0])
    J_max = len(D[0][0])
    K_max = len(D[0])

    D_hexx = [[0] * K_max * J_max * I_max * n for _ in range(group)]
    for g in range(group):
        for k in range(K_max):
            for j in range(J_max):
                for i in range(I_max):
                    m = k * (J_max * I_max) + j * I_max + i
                    for l in range(n):
                        D_hexx[g][m * n + l] = D[g][k][j][i]

    max_conv = max(conv_tri)
    D_hexx_reshaped = [[0] * max_conv for _ in range(group)]
    for g in range(group):
        for m in range(len(conv_tri)):
            if D_hexx[g][m] != 0:
                D_hexx_reshaped[g][conv_tri[m]-1] = D_hexx[g][m]

    # Extract boundary conditions for north, south, east, and west
    BC_north = BC[0]
    BC_south = BC[1]
    BC_east = BC[2]
    BC_west = BC[3]
    BC_top = BC[4]
    BC_bottom = BC[5]

    # Initialize the full matrix with zeros
    matrix = lil_matrix((group * max_conv, group * max_conv))

    # Loop through each group and each point in the conv array
    for g in range(group):
        for i in range(max_conv):
            idx = g * max_conv + i
            D_i = D_hexx_reshaped[g][i]

            # Loop through neighbors
            for neighbor_idx, neighbor in enumerate(conv_neighbor[i]):
                if neighbor_idx < 3:
                    if neighbor < 0:
                        # Handle boundary conditions for points with no neighbors
                        if neighbor == -1:
                            matrix[idx, idx] += DIFCOEFF_TRI_BC_plane(D_i, h, BC_north)
                        elif neighbor == -2:
                            matrix[idx, idx] += DIFCOEFF_TRI_BC_plane(D_i, h, BC_east)
                        elif neighbor == -3:
                            matrix[idx, idx] += DIFCOEFF_TRI_BC_plane(D_i, h, BC_south)
                        elif neighbor == -4:
                            matrix[idx, idx] += DIFCOEFF_TRI_BC_plane(D_i, h, BC_west)
                    else:
                        neighbor_conv_idx = g * max_conv + (neighbor)
                        D_next = D_hexx_reshaped[g][neighbor]
                        diff_coeff = DIFCOEFF_TRI_INTERIOR_plane(D_i, D_next, h)
                        matrix[idx, neighbor_conv_idx] += -diff_coeff
                        matrix[idx, idx] += diff_coeff
                else:
                    if neighbor == -1:
                        matrix[idx, idx] += DIFCOEFF_TRI_BC_axial(D_i, dz, BC_top)
#                        # Handle boundary conditions for points with no neighbors
#                        if neighbor == -5:
#                            matrix[idx, idx] += DIFCOEFF_TRI_BC_axial(D_i, dz, BC_top)
#                        elif neighbor == -6:
#                            matrix[idx, idx] += DIFCOEFF_TRI_BC_axial(D_i, dz, BC_bottom)
                    else:
                        neighbor_conv_idx = g * max_conv + (neighbor)
                        D_next = D_hexx_reshaped[g][neighbor]
                        diff_coeff = DIFCOEFF_TRI_INTERIOR_axial(D_i, D_next, dz)
                        matrix[idx, neighbor_conv_idx] += -diff_coeff
                        matrix[idx, idx] += diff_coeff

    print("D_hexx_mat generated")
    return matrix

def ADJOINT_TOT_3D_hexx_matrix(group, K_max, J_max, I_max, conv_tri, TOT, level):
    TOT_hexx = expand_XS_hexx_3D(group, K_max, J_max, I_max, TOT, level)

    max_conv = max(conv_tri)
    matrix = lil_matrix((group * max_conv, group * max_conv))
    for g in range(group):
        for n in range(len(TOT_hexx[0])):
            if TOT_hexx[g][n] != 0:
                matrix[g*max_conv+(conv_tri[n]-1), g*max_conv+(conv_tri[n]-1)] += TOT_hexx[g][n]
    print("TOT_mat generated")
    return matrix.transpose()

def ADJOINT_SCAT_3D_hexx_matrix(group, K_max, J_max, I_max, conv_tri, SIGS_reshaped, level):
    SIGS_hexx = expand_SIGS_hexx_3D(group, K_max, J_max, I_max, SIGS_reshaped, level)

    N = K_max * J_max * I_max
    max_conv = max(conv_tri)
    matrix = lil_matrix((group*max_conv, group*max_conv))
    if group == 1:
        for i in range(N):
            matrix[(conv_tri[k]-1), (conv_tri[k]-1)] += SIGS_hexx[0][i]
    else:
        for i in range(group):
            for j in range(group):
                for k in range(len(conv_tri)):
                    if SIGS_hexx[i][j][k] != 0:
                        matrix[i*max_conv+(conv_tri[k]-1), j*max_conv+(conv_tri[k]-1)] += SIGS_hexx[i][j][k]
    print("SCAT_mat generated")
    return matrix.transpose()

def ADJOINT_NUFIS_3D_hexx_matrix(group, K_max, J_max, I_max, conv_tri, chi, NUFIS, level):
    chi_hexx = expand_XS_hexx_3D(group, K_max, J_max, I_max, chi, level)
    NUFIS_hexx = expand_XS_hexx_3D(group, K_max, J_max, I_max, NUFIS, level)
    
    max_conv = max(conv_tri)
    matrix = lil_matrix((group * max_conv, group * max_conv))
    for i in range(group):
        for j in range(group):
            for k in range(len(NUFIS_hexx[0])):
                if NUFIS_hexx[j][k] != 0:
                    matrix[i*max_conv + (conv_tri[k]-1), j*max_conv + (conv_tri[k]-1)] += chi_hexx[i][k]*NUFIS_hexx[j][k]

    print("F_mat generated")
    return matrix.transpose()

##############################################################################
def NOISE_D_3D_hexx_matrix(group, BC, conv_tri, conv_neighbor, h, dz, D, level):
    def DIFCOEFF_TRI_INTERIOR_plane(D_i, D_next, h):
        return (8 / h**2) * ((D_i * D_next) / (D_i + D_next))

    def DIFCOEFF_TRI_BC_plane(D_i, h, BC):
        if BC == 1:  # Zero Flux
            return (8 * D_i) / ((h**2))
        if BC == 2:  # Reflective
            return 0
        if BC == 3:  # Vacuum
            return (8 * D_i) / ((4 * D_i * h * np.sqrt(3)) + (h**2))

    def DIFCOEFF_TRI_INTERIOR_axial(D_i, D_next, dz):
        return (2 / dz**2) * ((D_i * D_next) / (D_i + D_next))

    def DIFCOEFF_TRI_BC_axial(D_i, dz, BC):
        if BC == 1:  # Zero Flux
            return (2 * D_i) / ((dz**2))
        if BC == 2:  # Reflective
            return 0
        if BC == 3:  # Vacuum
            return (2 * D_i) / ((4 * D_i * dz) + (dz**2))

    n = 6 * (4 ** (level - 1))
    I_max = len(D[0][0][0])
    J_max = len(D[0][0])
    K_max = len(D[0])

    D_hexx = [[0] * K_max * J_max * I_max * n for _ in range(group)]
    for g in range(group):
        for k in range(K_max):
            for j in range(J_max):
                for i in range(I_max):
                    m = k * (J_max * I_max) + j * I_max + i
                    for l in range(n):
                        D_hexx[g][m * n + l] = D[g][k][j][i]

    max_conv = max(conv_tri)
    D_hexx_reshaped = [[0] * max_conv for _ in range(group)]
    for g in range(group):
        for m in range(len(conv_tri)):
            if D_hexx[g][m] != 0:
                D_hexx_reshaped[g][conv_tri[m]-1] = D_hexx[g][m]

    # Extract boundary conditions for north, south, east, and west
    BC_north = BC[0]
    BC_south = BC[1]
    BC_east = BC[2]
    BC_west = BC[3]
    BC_top = BC[4]
    BC_bottom = BC[5]

    # Initialize the full matrix with zeros
    matrix = lil_matrix((group * max_conv, group * max_conv))

    # Loop through each group and each point in the conv array
    for g in range(group):
        for i in range(max_conv):
            idx = g * max_conv + i
            D_i = D_hexx_reshaped[g][i]

            # Loop through neighbors
            for neighbor_idx, neighbor in enumerate(conv_neighbor[i]):
                if neighbor_idx < 3:
                    if neighbor < 0:
                        # Handle boundary conditions for points with no neighbors
                        if neighbor == -1:
                            neighbor_diff_coeff = DIFCOEFF_TRI_BC_plane(D_i, h, BC_north)
                            matrix[idx, idx] += -neighbor_diff_coeff
                        elif neighbor == -2:
                            neighbor_diff_coeff = DIFCOEFF_TRI_BC_plane(D_i, h, BC_east)
                            matrix[idx, idx] += -neighbor_diff_coeff
                        elif neighbor == -3:
                            neighbor_diff_coeff = DIFCOEFF_TRI_BC_plane(D_i, h, BC_south)
                            matrix[idx, idx] += -neighbor_diff_coeff
                        elif neighbor == -4:
                            neighbor_diff_coeff = DIFCOEFF_TRI_BC_plane(D_i, h, BC_west)
                            matrix[idx, idx] += -neighbor_diff_coeff
                    else:
                        neighbor_conv_idx = g * max_conv + (neighbor)
                        D_next = D_hexx_reshaped[g][neighbor]
                        diff_coeff = DIFCOEFF_TRI_INTERIOR_plane(D_i, D_next, h)
                        matrix[idx, neighbor_conv_idx] += diff_coeff
                        matrix[idx, idx] += -diff_coeff
                else:
                    if neighbor == -1:
                        neighbor_diff_coeff_axial = DIFCOEFF_TRI_BC_axial(D_i, dz, BC_top)
                        matrix[idx, idx] += -neighbor_diff_coeff_axial
#                    if neighbor < 0:
#                        # Handle boundary conditions for points with no neighbors
#                        if neighbor == -5:
#                            neighbor_diff_coeff = DIFCOEFF_TRI_BC_axial(D_i, h, BC_top)
#                            matrix[idx, idx] += -neighbor_diff_coeff
#                        elif neighbor == -6:
#                            neighbor_diff_coeff = DIFCOEFF_TRI_BC_axial(D_i, h, BC_bottom)
#                            matrix[idx, idx] += -neighbor_diff_coeff
                    else:
                        neighbor_conv_idx = g * max_conv + (neighbor)
                        D_next = D_hexx_reshaped[g][neighbor]
                        diff_coeff = DIFCOEFF_TRI_INTERIOR_axial(D_i, D_next, dz)
                        matrix[idx, neighbor_conv_idx] += diff_coeff
                        matrix[idx, idx] += -diff_coeff

    print("D_hexx_mat generated")
    return matrix

def NOISE_TOT_3D_hexx_matrix(group, K_max, J_max, I_max, conv_tri, TOT, level):
    TOT_hexx = expand_XS_hexx_3D(group, K_max, J_max, I_max, TOT, level)

    max_conv = max(conv_tri)
    matrix = lil_matrix((group * max_conv, group * max_conv))
    for g in range(group):
        for n in range(len(TOT_hexx[0])):
            if TOT_hexx[g][n] != 0:
                matrix[g*max_conv+(conv_tri[n]-1), g*max_conv+(conv_tri[n]-1)] += TOT_hexx[g][n]
    print("TOT_mat generated")
    return matrix

def NOISE_SCAT_3D_hexx_matrix(group, K_max, J_max, I_max, conv_tri, SIGS_reshaped, level):
    SIGS_hexx = expand_SIGS_hexx_3D(group, K_max, J_max, I_max, SIGS_reshaped, level)

    N = K_max * I_max * J_max
    max_conv = max(conv_tri)
    matrix = lil_matrix((group*max_conv, group*max_conv))
    if group == 1:
        for i in range(N):
            matrix[(conv_tri[k]-1), (conv_tri[k]-1)] += SIGS_hexx[0][i]
    else:
        for i in range(group):
            for j in range(group):
                for k in range(len(conv_tri)):
                    if SIGS_hexx[i][j][k] != 0:
                        matrix[i*max_conv+(conv_tri[k]-1), j*max_conv+(conv_tri[k]-1)] += SIGS_hexx[i][j][k]
    print("SCAT_mat generated")
    return matrix

def NOISE_NUFIS_3D_hexx_matrix(group, K_max, I_max, J_max, conv_tri, chi_p, chi_d, NUFIS, k_complex, Beff, keff, level):
    chi_p_hexx = expand_XS_hexx_3D(group, K_max, J_max, I_max, chi_p, level)
    chi_d_hexx = expand_XS_hexx_3D(group, K_max, J_max, I_max, chi_d, level)
    NUFIS_hexx = expand_XS_hexx_3D(group, K_max, J_max, I_max, NUFIS, level)
    
    max_conv = max(conv_tri)
    matrix = lil_matrix((group * max_conv, group * max_conv), dtype=complex)
    for i in range(group):
        for j in range(group):
            for k in range(len(NUFIS_hexx[0])):
                if NUFIS_hexx[j][k] != 0:
                    matrix[i*max_conv + (conv_tri[k]-1), j*max_conv + (conv_tri[k]-1)] += (chi_p_hexx[i][k] * (1-Beff)/keff + chi_d_hexx[i][k] * k_complex) * NUFIS_hexx[j][k]

    print("NUFIS_mat generated")
    return matrix

def NOISE_FREQ_3D_hexx_matrix(group, K_max, J_max, I_max, conv_tri, omega, v, level):
    v_hexx = expand_XS_hexx_3D(group, K_max, J_max, I_max, v, level)

    max_conv = max(conv_tri)
    matrix = lil_matrix((group * max_conv, group * max_conv), dtype=complex)
    for g in range(group):
        for n in range(len(conv_tri)):
            if v_hexx[g][n] != 0:
                matrix[g*max_conv+(conv_tri[n]-1), g*max_conv+(conv_tri[n]-1)] += 1j*omega/v_hexx[g][n]
    print("FREQ_mat generated")
    return matrix

def NOISE_dTOT_3D_hexx_matrix(group, K_max, J_max, I_max, conv_tri, dTOT_hexx, level):
    max_conv = max(conv_tri)
    matrix = lil_matrix((group * max_conv, group * max_conv), dtype=complex)
    for g in range(group):
        for n in range(len(dTOT_hexx[0])):
            if dTOT_hexx[g][n] != 0:
                matrix[g*max_conv+(conv_tri[n]-1), g*max_conv+(conv_tri[n]-1)] += dTOT_hexx[g][n]
    print("dTOT_mat generated")

#    dTOT_hex_temp = [[0] * max_conv for _ in range(group)]
#    for g in range(group):
#        for n in range(len(dTOT_hexx[0])):
#            dTOT_hex_temp[g][conv_tri[n]-1] = dTOT_hexx[g][n]
#            if dTOT_hexx[g][n] != 0:
#                print(f"dTOT_hexx_temp[{g}][{conv_tri[n]-1}] = {dTOT_hexx[g][n]}")

    return matrix

def NOISE_dSCAT_3D_hexx_matrix(group, K_max, J_max, I_max, conv_tri, dSIGS_hexx, level):
    N = K_max * I_max * J_max
    max_conv = max(conv_tri)
    matrix = lil_matrix((group*max_conv, group*max_conv), dtype=complex)
    if group == 1:
        for i in range(N):
            matrix[(conv_tri[k]-1), (conv_tri[k]-1)] += dSIGS_hexx[0][i]
    else:
        for i in range(group):
            for j in range(group):
                for k in range(len(conv_tri)):
                    if dSIGS_hexx[i][j][k] != 0:
                        matrix[i*max_conv+(conv_tri[k]-1), j*max_conv+(conv_tri[k]-1)] += dSIGS_hexx[i][j][k]
    print("SCAT_mat generated")
    return matrix

def NOISE_dNUFIS_3D_hexx_matrix(group, K_max, J_max, I_max, conv_tri, chi_p_hexx, chi_d_hexx, dNUFIS_hexx, k_complex, Beff, keff, level):
    max_conv = max(conv_tri)
    matrix = lil_matrix((group * max_conv, group * max_conv), dtype=complex)
    for i in range(group):
        for j in range(group):
            for k in range(len(dNUFIS_hexx[0])):
                if dNUFIS_hexx[j][k] != 0:
                    matrix[i*max_conv + (conv_tri[k]-1), j*max_conv + (conv_tri[k]-1)] += (chi_p_hexx[i][k] * (1-Beff)/keff + chi_d_hexx[i][k] * k_complex) * dNUFIS_hexx[j][k]

    print("dNUFIS_mat generated")
    return matrix

##############################################################################
def plot_triangular_3D(PHIg, x_coords, y_coords, k, tri_indices, g, cmap='viridis', varname=None, title=None, case_name=None, output_dir=None, solve=None, process_data=None):
    if process_data == 'magnitude':
        PHIg = np.abs(PHIg)  # Compute magnitude
    elif process_data == 'phase':
        PHIg_rad = np.angle(PHIg)  # Compute phase
        PHIg = np.degrees(PHIg_rad)  # Convert rad to deg
    
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    x = []
    y = []

    for i in range(len(x_coords)):
        x.append(x_coords[i]-x_center)
        y.append(y_coords[i]-y_center)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_aspect('equal')

    # Create the triangular color plot
    tri_plot = ax.tripcolor(x, y, tri_indices, facecolors=PHIg, cmap=cmap)

    # Add colorbar linked to the tri_plot object
    cbar_label = f'{varname}{g}'
    if solve == 'noise' and process_data == 'phase':
        cbar_label += '_deg'
    elif solve == 'noise' and process_data == 'magnitude':
        cbar_label += '_mag'

    cbar = fig.colorbar(tri_plot, ax=ax, label=cbar_label)
    
    if title:
        plt.title(title)

    # Note: 
    # solve could be:
    # "FORWARD", 
    # "ADJOINT", 
    # "NOISE", "NOISE_GREEN", "NOISE_UNFOLD", "NOISE_dPOWER", 
    # "NOISE_{position_noise}_{type_noise_str}", "NOISE_GREEN_{position_noise}_{type_noise_str}", "NOISE_UNFOLD_{position_noise}_{type_noise_str}", "NOISE_dPOWER_{position_noise}_{type_noise_str}", 
    filename = f'{output_dir}/{case_name}_{solve}/{case_name}_{solve}_{varname}_{process_data}_G{g}_Z{k}.png'
    plt.savefig(filename,
        bbox_inches='tight',
        pad_inches=0.05)
    plt.close(fig)

    return filename

def plot_triangular_3D_general(PHIg, x_coords, y_coords, k, tri_indices, g, cmap='viridis', varname=None, title=None, case_name=None, output_dir=None, process_data=None):
    if process_data == 'magnitude':
        PHIg = np.abs(PHIg)  # Compute magnitude
    elif process_data == 'phase':
        PHIg_rad = np.angle(PHIg)  # Compute phase
        PHIg = np.degrees(PHIg_rad)  # Convert rad to deg
    
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    x = []
    y = []

    for i in range(len(x_coords)):
        x.append(x_coords[i]-x_center)
        y.append(y_coords[i]-y_center)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_aspect('equal')

    # Create the triangular color plot
    tri_plot = ax.tripcolor(x, y, tri_indices, facecolors=PHIg, cmap=cmap)

    # Add colorbar linked to the tri_plot object
    cbar_label = f'{varname}{g}'
    if process_data == 'phase':
        cbar_label += '_deg'
    elif process_data == 'magnitude':
        cbar_label += '_mag'

    cbar = fig.colorbar(tri_plot, ax=ax, label=cbar_label)
    
    if title:
        plt.title(title)

    # Note: 
    # solve could be:
    # "FORWARD", 
    # "ADJOINT", 
    # "NOISE", "NOISE_GREEN", "NOISE_UNFOLD", "NOISE_dPOWER", 
    # "NOISE_{position_noise}_{type_noise_str}", "NOISE_GREEN_{position_noise}_{type_noise_str}", "NOISE_UNFOLD_{position_noise}_{type_noise_str}", "NOISE_dPOWER_{position_noise}_{type_noise_str}", 
    filename = f'{output_dir}_{varname}_{process_data}_G{g}_Z{k}.png'
    plt.savefig(filename,
        bbox_inches='tight',
        pad_inches=0.05)
    plt.close(fig)

    return filename

def plot_triangular_3D_categorical(PHIg, x_coords, y_coords, k, tri_indices, g, varname=None, title=None, case_name=None, output_dir=None):
   
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    x = []
    y = []

    for i in range(len(x_coords)):
        x.append(x_coords[i]-x_center)
        y.append(y_coords[i]-y_center)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_aspect('equal')

    # Define categories and corresponding colors
    categories = [0, 0.5, 1, 1.5]
    colors = ['lightgray', 'blue', 'red', 'purple']
    labels = ['Empty', 'Detector', 'Source', 'Detector + Source']

    # Create a colormap for discrete values
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm([c-0.25 for c in categories] + [categories[-1]+0.25], cmap.N)

    # Create the triangular color plot
    tri_plot = ax.tripcolor(x, y, tri_indices, facecolors=PHIg, cmap=cmap, norm=norm)

    # Create a legend manually
    legend_elements = [Patch(facecolor=colors[i], edgecolor='k', label=labels[i]) for i in range(len(categories))]
    ax.legend(handles=legend_elements,
               loc='upper center',          # center it
               bbox_to_anchor=(0.5, -0.12), # shift below the plot
               ncol=len(categories),        # put all items in one row
               frameon=False)               # remove box around legend
    
    if title:
        plt.title(title)

    # Note: 
    # solve could be:
    # "FORWARD", 
    # "ADJOINT", 
    # "NOISE", "NOISE_GREEN", "NOISE_UNFOLD", "NOISE_dPOWER", 
    # "NOISE_{position_noise}_{type_noise_str}", "NOISE_GREEN_{position_noise}_{type_noise_str}", "NOISE_UNFOLD_{position_noise}_{type_noise_str}", "NOISE_dPOWER_{position_noise}_{type_noise_str}", 
    filename = f'{output_dir}_{varname}_categorical_G{g}_Z{k}.png'
    plt.savefig(filename,
        bbox_inches='tight',
        pad_inches=0.05)
    plt.close(fig)

    return filename

##############################################################################
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

def find_boundary_for_each_hex(level, triangle_neighbors, ownership):
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

    # Loop through the boundary triangles and classify vertical boundaries
    for hex_idx, boundary in hex_boundaries.items():
        for i, tri_idx in enumerate(boundary):
            if (i // 2**(level-1) == 2) or (i // 2**(level-1) == 5):
                hex_vert_boundaries[hex_idx].append(tri_idx)

    return hex_boundaries, hex_vert_boundaries

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

def XS3D_FXV(level, group, s, K_max, J_max, I_max, dTOT, dNUFIS, fav_strength, diff_X_ABS, diff_X_NUFIS):
    dTOT_hexx = expand_XS_hexx_3D(group, K_max, J_max, I_max, dTOT, level)
    dNUFIS_hexx = expand_XS_hexx_3D(group, K_max, J_max, I_max, dNUFIS, level)

    p = 6 * (4 ** (level - 1))
    boundary_tris_new = reference_hex_boundary_triangles(s, level, threshold=1.0)[1]

    # Calculate the size of each region (assuming the length is divisible by 6)
    region_size = len(boundary_tris_new) // 6

    # Define the 6 regions
    northeast_hexx_boundary = boundary_tris_new[:region_size]
    northwest_hexx_boundary = boundary_tris_new[region_size:2*region_size]
    west_hexx_boundary = boundary_tris_new[2*region_size:3*region_size]
    southwest_hexx_boundary = boundary_tris_new[3*region_size:4*region_size]
    southeast_hexx_boundary = boundary_tris_new[4*region_size:5*region_size]
    east_hexx_boundary = boundary_tris_new[5*region_size:]

    # Note: the commented part will be useful is epsilon/d model is used
    for g in range(group):
        for m in range(K_max * J_max * I_max):
            k = m // (J_max * I_max)
            if dTOT[g][m] != 0:
                for t in range(p):
                    if t not in east_hexx_boundary and t not in west_hexx_boundary:
                        dTOT_hexx[g][m * p + t] = 0
                        dNUFIS_hexx[g][m * p + t] = 0
                    for t_idx, t in enumerate(east_hexx_boundary):
                        dTOT_hexx[g][m * p + t] = fav_strength * diff_X_ABS[g][0][k]
                        dTOT_hexx[g][(m+1) * p + west_hexx_boundary[t_idx]] = fav_strength * diff_X_ABS[g][0][k]
                        dNUFIS_hexx[g][m * p + t] = fav_strength * diff_X_NUFIS[g][0][k]
                        dNUFIS_hexx[g][(m+1) * p + west_hexx_boundary[t_idx]] = fav_strength * diff_X_NUFIS[g][0][k]
                    for t_idx, t in enumerate(west_hexx_boundary):
                        dTOT_hexx[g][m * p + t] = fav_strength * diff_X_ABS[g][1][k]
                        dTOT_hexx[g][(m-1) * p + east_hexx_boundary[t_idx]] = fav_strength * diff_X_ABS[g][1][k]
                        dNUFIS_hexx[g][m * p + t] = fav_strength * diff_X_NUFIS[g][1][k]
                        dNUFIS_hexx[g][(m-1) * p + east_hexx_boundary[t_idx]] = fav_strength * diff_X_NUFIS[g][1][k]

    return dTOT_hexx, dNUFIS_hexx

def XS3D_FAV(level, group, s, K_max, J_max, I_max, dTOT, dNUFIS, fav_strength, diff_X_ABS, diff_X_NUFIS):
    dTOT_hexx = expand_XS_hexx_3D(group, J_max, I_max, K_max, dTOT, level)
    dNUFIS_hexx = expand_XS_hexx_3D(group, J_max, I_max, K_max, dNUFIS, level)

    p = 6 * (4 ** (level - 1))
    boundary_tris_new = reference_hex_boundary_triangles(s, level, threshold=1.0)[1]

    # Calculate the size of each region (assuming the length is divisible by 6)
    region_size = len(boundary_tris_new) // 6

    # Define the 6 regions
    northeast_hexx_boundary = boundary_tris_new[:region_size]
    northwest_hexx_boundary = boundary_tris_new[region_size:2*region_size]
    west_hexx_boundary = boundary_tris_new[2*region_size:3*region_size]
    southwest_hexx_boundary = boundary_tris_new[3*region_size:4*region_size]
    southeast_hexx_boundary = boundary_tris_new[4*region_size:5*region_size]
    east_hexx_boundary = boundary_tris_new[5*region_size:]

    # Note: the commented part will be useful is epsilon/d model is used
    for g in range(group):
        for m in range(K_max * J_max * I_max):
            k = m // (J_max * I_max)
            if dTOT[g][m] != 0:
                for t in range(p):
                    if t not in east_hexx_boundary and t not in west_hexx_boundary and t not in northeast_hexx_boundary and t not in northwest_hexx_boundary and t not in southeast_hexx_boundary and t not in southwest_hexx_boundary:
                        dTOT_hexx[g][m * p + t] = 0
                        dNUFIS_hexx[g][m * p + t] = 0
                    for t_idx, t in enumerate(east_hexx_boundary):
                        dTOT_hexx[g][m * p + (t)] = fav_strength * diff_X_ABS[g][0][k]
                        dTOT_hexx[g][(m+1) * p + (west_hexx_boundary[t_idx])] = fav_strength * diff_X_ABS[g][0][k]
                        dNUFIS_hexx[g][m * p + (t)] = fav_strength * diff_X_NUFIS[g][0][k]
                        dNUFIS_hexx[g][(m+1) * p + (west_hexx_boundary[t_idx])] = fav_strength * diff_X_NUFIS[g][0][k]
                    for t_idx, t in enumerate(west_hexx_boundary):
                        dTOT_hexx[g][m * p + (t)] = fav_strength * diff_X_ABS[g][1][k]
                        dTOT_hexx[g][(m-1) * p + (east_hexx_boundary[t_idx])] = fav_strength * diff_X_ABS[g][1][k]
                        dNUFIS_hexx[g][m * p + (t)] = fav_strength * diff_X_NUFIS[g][1][k]
                        dNUFIS_hexx[g][(m-1) * p + (east_hexx_boundary[t_idx])] = fav_strength * diff_X_NUFIS[g][1][k]
                    for t_idx, t in enumerate(northeast_hexx_boundary):
                        dTOT_hexx[g][m * p + (t)] = fav_strength * diff_X_ABS[g][2][k]
                        dTOT_hexx[g][(m + I_max) * p + (southwest_hexx_boundary[t_idx])] = fav_strength * diff_X_ABS[g][2][k]
                        dNUFIS_hexx[g][m * p + (t)] = fav_strength * diff_X_NUFIS[g][2][k]
                        dNUFIS_hexx[g][(m + I_max) * p + (southwest_hexx_boundary[t_idx])] = fav_strength * diff_X_NUFIS[g][2][k]
                    for t_idx, t in enumerate(northwest_hexx_boundary):
                        dTOT_hexx[g][m * p + (t)] = fav_strength * diff_X_ABS[g][3][k]
                        dTOT_hexx[g][(m + I_max - 1) * p + (southeast_hexx_boundary[t_idx])] = fav_strength * diff_X_ABS[g][3][k]
                        dNUFIS_hexx[g][m * p + (t)] = fav_strength * diff_X_NUFIS[g][3][k]
                        dNUFIS_hexx[g][(m + I_max - 1) * p + (southeast_hexx_boundary[t_idx])] = fav_strength * diff_X_NUFIS[g][3][k]
                    for t_idx, t in enumerate(southeast_hexx_boundary):
                        dTOT_hexx[g][m * p + (t)] = fav_strength * diff_X_ABS[g][4][k]
                        dTOT_hexx[g][(m - I_max + 1) * p + (northwest_hexx_boundary[t_idx])] = fav_strength * diff_X_ABS[g][4][k]
                        dNUFIS_hexx[g][m * p + (t)] = fav_strength * diff_X_NUFIS[g][4][k]
                        dNUFIS_hexx[g][(m - I_max + 1) * p + (northwest_hexx_boundary[t_idx])] = fav_strength * diff_X_NUFIS[g][4][k]
                    for t_idx, t in enumerate(southwest_hexx_boundary):
                        dTOT_hexx[g][m * p + (t)] = fav_strength * diff_X_ABS[g][5][k]
                        dTOT_hexx[g][(m - I_max) * p + (northeast_hexx_boundary[t_idx])] = fav_strength * diff_X_ABS[g][5][k]
                        dNUFIS_hexx[g][m * p + (t)] = fav_strength * diff_X_NUFIS[g][5][k]
                        dNUFIS_hexx[g][(m - I_max) * p + (northeast_hexx_boundary[t_idx])] = fav_strength * diff_X_NUFIS[g][5][k]

    return dTOT_hexx, dNUFIS_hexx

def interpolate_3D_hexx_rbf(dPHI_zero, group, K_max, conv_tri, known_coords, known_values_group, zero_coords, all_triangles, zero_coord_to_index):
    if len(dPHI_zero) == group * max(conv_tri):
        dPHI_zero_new = np.reshape(dPHI_zero, (group, K_max, len(all_triangles)))
    else:
        dPHI_zero_new = dPHI_zero
    dPHI_interp = dPHI_zero_new.copy()

    for g in range(group):
        for k in range(K_max):
            PHIg_temp = np.zeros(len(all_triangles), dtype=complex)
            if zero_coords:
                try:
                    rbf_interpolator = RBFInterpolator(known_coords[k], known_values_group[g][k], kernel='thin_plate_spline')
                    interpolated_values = rbf_interpolator(zero_coords[k])

                    # Use precomputed mapping instead of recalculating centroids
                    for coord, value in zip(zero_coords[k], interpolated_values):
                        if coord in zero_coord_to_index:
                            PHIg_temp[zero_coord_to_index[coord]] = value
                except Exception as e:
                    print(f"RBF Interpolation failed for group {g}: {e}")
                    continue

            dPHI_interp[g][k] = PHIg_temp

    # Step 4: Reformat results if needed
    if len(dPHI_zero) == group * max(conv_tri):
        dPHI_interp_new = np.zeros((group * K_max * len(all_triangles)), dtype=complex)
        for g in range(group):
            for k in range(K_max):
                for n in range(len(all_triangles)):
                    dPHI_interp_new[g * max(conv_tri) + k * len(all_triangles) + n] = dPHI_interp[g][k][n]
    else:
        dPHI_interp_new = dPHI_interp

    return dPHI_interp_new
