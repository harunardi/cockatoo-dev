import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csc_matrix
import os
import h5py
import sys
from scipy.interpolate import griddata
from scipy.interpolate import RBFInterpolator
from matplotlib.colors import Normalize
from matplotlib import cm
from PIL import Image
from scipy.spatial.distance import pdist, squareform
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

# Prevent .pyc file generation
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True

#######################################################################################################
# DEFAULT VALUES
output_dir = None
case_name = None
x, y = 0, 0

#######################################################################################################
# Function to convert 3D rectangular indexes
def convert_index_3D_rect(D, I_max, J_max, K_max):
    conv = [0] * (I_max * J_max * K_max)
    tmp_conv = 0
    for k in range(K_max):
        for j in range(J_max):
            for i in range(I_max):
                if D[0][k][j][i] != 0:
                    tmp_conv += 1
                    m = k * (I_max * J_max) + j * I_max + i
                    conv[m] = tmp_conv
    return conv

# Function to save sparse matrix to file
def save_sparse_matrix(A, filename):
    # Convert lil_matrix to coo_matrix to use find() function
    A_coo = A.tocoo()
    
    # Extract row indices, column indices, and values of non-zero elements
    I, J, V = A_coo.row, A_coo.col, A_coo.data
    
    # Write data to file
    with open(filename, 'w') as file:
        for i, j, v in zip(I, J, V):
            file.write(f"{i} {j} {v}\n")
    
    print(f"Sparse matrix saved to {filename}")

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
def FORWARD_D_3D_rect_matrix(group, BC, conv, dx, dy, dz, D):
    def DIFXCOEF(D_west, D_mid, D_east, dx):
        a1 = (2*D_mid*D_west)/((D_west+D_mid)*dx**2)
        a2 = (2*D_mid*D_west)/((D_west+D_mid)*dx**2) + (2*D_east*D_mid)/((D_east+D_mid)*dx**2)
        a3 = (2*D_east*D_mid)/((D_east+D_mid)*dx**2)
        return a1, a2, a3
    
    def DIFXCOEF_WB(D_mid, D_east, dx, BC_west):
        if BC_west == 1:  # Zero Flux
            a2 = (2*D_mid)/(dx**2) + (2*D_east*D_mid)/((D_east+D_mid)*dx**2)
        elif BC_west == 2:  # Reflective
            a2 = (2*D_east*D_mid)/((D_east+D_mid)*dx**2)
        elif BC_west == 3:  # Vacuum
            a2 = (2*D_mid)/((4*D_mid*dx)+dx**2) + (2*D_east*D_mid)/((D_east+D_mid)*dx**2)

        a3 = (2*D_east*D_mid)/((D_east+D_mid)*dx**2)
        return a2, a3
    
    def DIFXCOEF_EB(D_west, D_mid, dx, BC_east):
        if BC_east == 1:  # Zero Flux
            a2 = (2*D_mid)/(dx**2) + (2*D_west*D_mid)/((D_west+D_mid)*dx**2)
        elif BC_east == 2:  # Reflective
            a2 = (2*D_west*D_mid)/((D_west+D_mid)*dx**2)
        elif BC_east == 3:  # Vacuum
            a2 = (2*D_mid)/((4*D_mid*dx)+dx**2) + (2*D_mid*D_west)/((D_west+D_mid)*dx**2)

        a1 = (2*D_mid*D_west)/((D_west+D_mid)*dx**2)
        return a1, a2

    def DIFYCOEF(D_bot, D_mid, D_top, dy):
        b1 = (2*D_mid*D_bot)/((D_bot+D_mid)*dy**2)
        b2 = (2*D_mid*D_bot)/((D_bot+D_mid)*dy**2) + (2*D_top*D_mid)/((D_top+D_mid)*dy**2)
        b3 = (2*D_top*D_mid)/((D_top+D_mid)*dy**2)
        return b1, b2, b3

    def DIFYCOEF_SB(D_mid, D_top, dy, BC_south):
        if BC_south == 1:  # Zero Flux
            b2 = (2*D_mid)/(dy**2) + (2*D_top*D_mid)/((D_top+D_mid)*dy**2)
        elif BC_south == 2:  # Reflective
            b2 = (2*D_top*D_mid)/((D_top+D_mid)*dy**2)
        elif BC_south == 3:  # Vacuum
            b2 = (2*D_mid)/((4*D_mid*dy)+dy**2) + (2*D_top*D_mid)/((D_top+D_mid)*dy**2)

        b3 = (2*D_top*D_mid)/((D_top+D_mid)*dy**2)
        return b2, b3

    def DIFYCOEF_NB(D_bot, D_mid, dy, BC_north):
        if BC_north == 1:  # Zero Flux
            b2 = (2*D_mid)/(dy**2) + (2*D_mid*D_bot)/((D_bot+D_mid)*dy**2)
        elif BC_north == 2:  # Reflective
            b2 = (2*D_mid*D_bot)/((D_bot+D_mid)*dy**2)
        elif BC_north == 3:  # Vacuum
            b2 = (2*D_mid)/((4*D_mid*dy)+dy**2) + (2*D_mid*D_bot)/((D_bot+D_mid)*dy**2)

        b1 = (2*D_mid*D_bot)/((D_bot+D_mid)*dy**2)
        return b1, b2

    def DIFZCOEF(D_bot, D_mid, D_top, dz):
        c1 = (2*D_mid*D_bot)/((D_bot+D_mid)*dz**2)
        c2 = (2*D_mid*D_bot)/((D_bot+D_mid)*dz**2) + (2*D_top*D_mid)/((D_top+D_mid)*dz**2)
        c3 = (2*D_top*D_mid)/((D_top+D_mid)*dz**2)
        return c1, c2, c3

    def DIFZCOEF_BB(D_mid, D_top, dz, BC_down):
        if BC_down == 1:  # Zero Flux
            c2 = (2*D_mid)/(dz**2) + (2*D_top*D_mid)/((D_top+D_mid)*dz**2)
        elif BC_down == 2:  # Reflective
            c2 = (2*D_top*D_mid)/((D_top+D_mid)*dz**2)
        elif BC_down == 3:  # Vacuum
            c2 = (2*D_mid)/((4*D_mid*dz)+dz**2) + (2*D_top*D_mid)/((D_top+D_mid)*dz**2)

        c3 = (2*D_top*D_mid)/((D_top+D_mid)*dz**2)
        return c2, c3

    def DIFZCOEF_TB(D_bot, D_mid, dz, BC_up):
        if BC_up == 1:  # Zero Flux
            c2 = (2*D_mid)/(dz**2) + (2*D_bot*D_mid)/((D_bot+D_mid)*dz**2)
        elif BC_up == 2:  # Reflective
            c2 = (2*D_bot*D_mid)/((D_bot+D_mid)*dz**2)
        elif BC_up == 3:  # Vacuum
            c2 = (2*D_mid)/((4*D_mid*dz)+dz**2) + (2*D_mid*D_bot)/((D_bot+D_mid)*dz**2)

        c1 = (2*D_mid*D_bot)/((D_bot+D_mid)*dz**2)
        return c1, c2
    
    # Initialize the full matrix with zeros
    BC_north = BC[0]
    BC_south = BC[1]
    BC_east = BC[2]
    BC_west = BC[3]
    BC_top = BC[4]
    BC_bottom = BC[5]
    I_max = len(D[0][0][0])  # N row
    J_max = len(D[0][0])  # N column
    K_max = len(D[0])  # N column
    max_conv = max(conv)
    matrix = lil_matrix((group*max_conv, group*max_conv))

    # Build DX
    DX = lil_matrix((group*max_conv, group*max_conv))
    for g in range(group):
        for k in range(K_max):  # Loop over depth
            for j in range(J_max):  # Loop over column
                for i in range(I_max):  # Loop over row
                    m = (k * J_max * I_max) + (j * I_max) + i
                    if D[g][k][j][i] != 0:
                        if i == 0 or (i > 0 and D[g][k][j][i-1] == 0):
                            a2, a3 = DIFXCOEF_WB(D[g][k][j][i], D[g][k][j][i+1], dx, BC_west)
                            DX[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m]-1)] += a2
                            if i < I_max-1:
                                DX[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m+1]-1)] += -a3
                        elif i == I_max-1 or (i < I_max-1 and D[g][k][j][i+1] == 0):
                            a1, a2 = DIFXCOEF_EB(D[g][k][j][i-1], D[g][k][j][i], dx, BC_east)
                            DX[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m]-1)] += a2
                            if i > 0:
                                DX[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m-1]-1)] += -a1
                        else:
                            a1, a2, a3 = DIFXCOEF(D[g][k][j][i-1], D[g][k][j][i], D[g][k][j][i+1], dx)
                            if i > 0:
                                DX[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m-1]-1)] += -a1
                            DX[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m]-1)] += a2
                            if i < I_max-1:
                                DX[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m+1]-1)] += -a3

    # Build DY
    DY = lil_matrix((group*max_conv, group*max_conv))
    for g in range(group):
        for k in range(K_max):  # Loop over depth
            for j in range(J_max):  # Loop over column
                for i in range(I_max):  # Loop over row
                    m = (k * J_max * I_max) + (j * I_max) + i
                    if D[g][k][j][i] != 0:
                        if j == 0 or (j > 0 and D[g][k][j-1][i] == 0):
                            b2, b3 = DIFYCOEF_SB(D[g][k][j][i], D[g][k][j+1][i], dy, BC_south)
                            DY[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m]-1)] += b2
                            if j < J_max-1:
                                DY[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m+I_max]-1)] += -b3
                        elif j == J_max-1 or (j < J_max-1 and D[g][k][j+1][i] == 0):
                            b1, b2 = DIFYCOEF_NB(D[g][k][j-1][i], D[g][k][j][i], dy, BC_north)
                            DY[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m]-1)] += b2
                            if j > 0:
                                DY[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m-I_max]-1)] += -b1
                        else:
                            b1, b2, b3 = DIFYCOEF(D[g][k][j-1][i], D[g][k][j][i], D[g][k][j+1][i], dy)
                            if j > 0:
                                DY[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m-I_max]-1)] += -b1
                            DY[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m]-1)] += b2
                            if j < J_max-1:
                                DY[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m+I_max]-1)] += -b3

    # Build DZ
    DZ = lil_matrix((group*max_conv, group*max_conv))
    for g in range(group):
        for k in range(K_max):  # Loop over depth
            for j in range(J_max):  # Loop over column
                for i in range(I_max):  # Loop over row
                    m = (k * J_max * I_max) + (j * I_max) + i
                    if D[g][k][j][i] != 0:
                        if k == 0 or (k > 0 and D[g][k-1][j][i] == 0):
                            c2, c3 = DIFZCOEF_BB(D[g][k][j][i], D[g][k+1][j][i], dz, BC_bottom)
                            DZ[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m]-1)] += c2
                            if k < K_max-1:
                                DZ[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m+(I_max*J_max)]-1)] += -c3
                        elif k == K_max-1 or (k < K_max-1 and D[g][k+1][j][i] == 0):
                            c1, c2 = DIFZCOEF_TB(D[g][k-1][j][i], D[g][k][j][i], dz, BC_top)
                            DZ[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m]-1)] += c2
                            if k > 0:
                                DZ[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m-(I_max*J_max)]-1)] += -c1
                        else:
                            c1, c2, c3 = DIFZCOEF(D[g][k-1][j][i], D[g][k][j][i], D[g][k+1][j][i], dz)
                            if k > 0:
                                DZ[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m-(I_max*J_max)]-1)] += -c1
                            DZ[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m]-1)] += c2
                            if k < K_max-1:
                                DZ[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m+(I_max*J_max)]-1)] += -c3

    matrix = DX + DY + DZ
    print("D_mat generated")
    return matrix

def FORWARD_TOT_3D_rect_matrix(g, N, conv, TOT):
    max_conv = max(conv)
    matrix = lil_matrix((g*max_conv, g*max_conv))
    for group in range(g):
        for n in range(N):
            matrix[group*max_conv+(conv[n]-1), group*max_conv+(conv[n]-1)] += TOT[group][n]
    print("TOT_mat generated")
    return matrix

def FORWARD_SCAT_3D_rect_matrix(g, N, conv, SIGS):
    max_conv = max(conv)
    matrix = lil_matrix((g*max_conv, g*max_conv))
    if g == 1:
        for i in range(N):
            matrix[(conv[k]-1), (conv[k]-1)] += SIGS[0][i]
    else:
        for i in range(g):
            for j in range(g):
                for k in range(N):
                    matrix[i*max_conv + (conv[k]-1), j*max_conv + (conv[k]-1)] += SIGS[i][j][k]
    print("SCAT_mat generated")
    return matrix

def FORWARD_NUFIS_3D_rect_matrix(g, N, conv, chi, NUFIS):
    max_conv = max(conv)
    matrix = lil_matrix((g*max_conv,g*max_conv))
    for i in range(g):
        for j in range(g):
            for k in range(N):
                    matrix[i*max_conv + (conv[k]-1), j*max_conv + (conv[k]-1)] += chi[i][k]*NUFIS[j][k]
    print("NUFIS_mat generated")
    return matrix

##############################################################################
def ADJOINT_D_3D_rect_matrix(group, BC, conv, dx, dy, dz, D):
    def DIFXCOEF(D_west, D_mid, D_east, dx):
        a1 = (2*D_mid*D_west)/((D_west+D_mid)*dx**2)
        a2 = (2*D_mid*D_west)/((D_west+D_mid)*dx**2) + (2*D_east*D_mid)/((D_east+D_mid)*dx**2)
        a3 = (2*D_east*D_mid)/((D_east+D_mid)*dx**2)
        return a1, a2, a3
    
    def DIFXCOEF_WB(D_mid, D_east, dx, BC_west):
        if BC_west == 1:  # Zero Flux
            a2 = (2*D_mid)/(dx**2) + (2*D_east*D_mid)/((D_east+D_mid)*dx**2)
        elif BC_west == 2:  # Reflective
            a2 = (2*D_east*D_mid)/((D_east+D_mid)*dx**2)
        elif BC_west == 3:  # Vacuum
            a2 = (2*D_mid)/((4*D_mid*dx)+dx**2) + (2*D_east*D_mid)/((D_east+D_mid)*dx**2)

        a3 = (2*D_east*D_mid)/((D_east+D_mid)*dx**2)
        return a2, a3
    
    def DIFXCOEF_EB(D_west, D_mid, dx, BC_east):
        if BC_east == 1:  # Zero Flux
            a2 = (2*D_mid)/(dx**2) + (2*D_west*D_mid)/((D_west+D_mid)*dx**2)
        elif BC_east == 2:  # Reflective
            a2 = (2*D_west*D_mid)/((D_west+D_mid)*dx**2)
        elif BC_east == 3:  # Vacuum
            a2 = (2*D_mid)/((4*D_mid*dx)+dx**2) + (2*D_mid*D_west)/((D_west+D_mid)*dx**2)

        a1 = (2*D_mid*D_west)/((D_west+D_mid)*dx**2)
        return a1, a2

    def DIFYCOEF(D_bot, D_mid, D_top, dy):
        b1 = (2*D_mid*D_bot)/((D_bot+D_mid)*dy**2)
        b2 = (2*D_mid*D_bot)/((D_bot+D_mid)*dy**2) + (2*D_top*D_mid)/((D_top+D_mid)*dy**2)
        b3 = (2*D_top*D_mid)/((D_top+D_mid)*dy**2)
        return b1, b2, b3

    def DIFYCOEF_SB(D_mid, D_top, dy, BC_south):
        if BC_south == 1:  # Zero Flux
            b2 = (2*D_mid)/(dy**2) + (2*D_top*D_mid)/((D_top+D_mid)*dy**2)
        elif BC_south == 2:  # Reflective
            b2 = (2*D_top*D_mid)/((D_top+D_mid)*dy**2)
        elif BC_south == 3:  # Vacuum
            b2 = (2*D_mid)/((4*D_mid*dy)+dy**2) + (2*D_top*D_mid)/((D_top+D_mid)*dy**2)

        b3 = (2*D_top*D_mid)/((D_top+D_mid)*dy**2)
        return b2, b3

    def DIFYCOEF_NB(D_bot, D_mid, dy, BC_north):
        if BC_north == 1:  # Zero Flux
            b2 = (2*D_mid)/(dy**2) + (2*D_mid*D_bot)/((D_bot+D_mid)*dy**2)
        elif BC_north == 2:  # Reflective
            b2 = (2*D_mid*D_bot)/((D_bot+D_mid)*dy**2)
        elif BC_north == 3:  # Vacuum
            b2 = (2*D_mid)/((4*D_mid*dy)+dy**2) + (2*D_mid*D_bot)/((D_bot+D_mid)*dy**2)

        b1 = (2*D_mid*D_bot)/((D_bot+D_mid)*dy**2)
        return b1, b2

    def DIFZCOEF(D_bot, D_mid, D_top, dz):
        c1 = (2*D_mid*D_bot)/((D_bot+D_mid)*dz**2)
        c2 = (2*D_mid*D_bot)/((D_bot+D_mid)*dz**2) + (2*D_top*D_mid)/((D_top+D_mid)*dz**2)
        c3 = (2*D_top*D_mid)/((D_top+D_mid)*dz**2)
        return c1, c2, c3

    def DIFZCOEF_BB(D_mid, D_top, dz, BC_down):
        if BC_down == 1:  # Zero Flux
            c2 = (2*D_mid)/(dz**2) + (2*D_top*D_mid)/((D_top+D_mid)*dz**2)
        elif BC_down == 2:  # Reflective
            c2 = (2*D_top*D_mid)/((D_top+D_mid)*dz**2)
        elif BC_down == 3:  # Vacuum
            c2 = (2*D_mid)/((4*D_mid*dz)+dz**2) + (2*D_top*D_mid)/((D_top+D_mid)*dz**2)

        c3 = (2*D_top*D_mid)/((D_top+D_mid)*dz**2)
        return c2, c3

    def DIFZCOEF_TB(D_bot, D_mid, dz, BC_up):
        if BC_up == 1:  # Zero Flux
            c2 = (2*D_mid)/(dz**2) + (2*D_bot*D_mid)/((D_bot+D_mid)*dz**2)
        elif BC_up == 2:  # Reflective
            c2 = (2*D_bot*D_mid)/((D_bot+D_mid)*dz**2)
        elif BC_up == 3:  # Vacuum
            c2 = (2*D_mid)/((4*D_mid*dz)+dz**2) + (2*D_mid*D_bot)/((D_bot+D_mid)*dz**2)

        c1 = (2*D_mid*D_bot)/((D_bot+D_mid)*dz**2)
        return c1, c2
    
    # Initialize the full matrix with zeros
    BC_north = BC[0]
    BC_south = BC[1]
    BC_east = BC[2]
    BC_west = BC[3]
    BC_top = BC[4]
    BC_bottom = BC[5]
    I_max = len(D[0][0][0])  # N row
    J_max = len(D[0][0])  # N column
    K_max = len(D[0])  # N column
    max_conv = max(conv)
    matrix = lil_matrix((group*max_conv, group*max_conv))

    # Build DX
    DX = lil_matrix((group*max_conv, group*max_conv))
    for g in range(group):
        for k in range(K_max):  # Loop over depth
            for j in range(J_max):  # Loop over column
                for i in range(I_max):  # Loop over row
                    m = (k * J_max * I_max) + (j * I_max) + i
                    if D[g][k][j][i] != 0:
                        if i == 0 or (i > 0 and D[g][k][j][i-1] == 0):
                            a2, a3 = DIFXCOEF_WB(D[g][k][j][i], D[g][k][j][i+1], dx, BC_west)
                            DX[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m]-1)] += a2
                            if i < I_max-1:
                                DX[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m+1]-1)] += -a3
                        elif i == I_max-1 or (i < I_max-1 and D[g][k][j][i+1] == 0):
                            a1, a2 = DIFXCOEF_EB(D[g][k][j][i-1], D[g][k][j][i], dx, BC_east)
                            DX[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m]-1)] += a2
                            if i > 0:
                                DX[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m-1]-1)] += -a1
                        else:
                            a1, a2, a3 = DIFXCOEF(D[g][k][j][i-1], D[g][k][j][i], D[g][k][j][i+1], dx)
                            if i > 0:
                                DX[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m-1]-1)] += -a1
                            DX[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m]-1)] += a2
                            if i < I_max-1:
                                DX[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m+1]-1)] += -a3

    # Build DY
    DY = lil_matrix((group*max_conv, group*max_conv))
    for g in range(group):
        for k in range(K_max):  # Loop over depth
            for j in range(J_max):  # Loop over column
                for i in range(I_max):  # Loop over row
                    m = (k * J_max * I_max) + (j * I_max) + i
                    if D[g][k][j][i] != 0:
                        if j == 0 or (j > 0 and D[g][k][j-1][i] == 0):
                            b2, b3 = DIFYCOEF_SB(D[g][k][j][i], D[g][k][j+1][i], dy, BC_south)
                            DY[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m]-1)] += b2
                            if j < J_max-1:
                                DY[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m+I_max]-1)] += -b3
                        elif j == J_max-1 or (j < J_max-1 and D[g][k][j+1][i] == 0):
                            b1, b2 = DIFYCOEF_NB(D[g][k][j-1][i], D[g][k][j][i], dy, BC_north)
                            DY[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m]-1)] += b2
                            if j > 0:
                                DY[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m-I_max]-1)] += -b1
                        else:
                            b1, b2, b3 = DIFYCOEF(D[g][k][j-1][i], D[g][k][j][i], D[g][k][j+1][i], dy)
                            if j > 0:
                                DY[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m-I_max]-1)] += -b1
                            DY[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m]-1)] += b2
                            if j < J_max-1:
                                DY[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m+I_max]-1)] += -b3

    # Build DZ
    DZ = lil_matrix((group*max_conv, group*max_conv))
    for g in range(group):
        for k in range(K_max):  # Loop over depth
            for j in range(J_max):  # Loop over column
                for i in range(I_max):  # Loop over row
                    m = (k * J_max * I_max) + (j * I_max) + i
                    if D[g][k][j][i] != 0:
                        if k == 0 or (k > 0 and D[g][k-1][j][i] == 0):
                            c2, c3 = DIFZCOEF_BB(D[g][k][j][i], D[g][k+1][j][i], dz, BC_bottom)
                            DZ[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m]-1)] += c2
                            if k < K_max-1:
                                DZ[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m+(I_max*J_max)]-1)] += -c3
                        elif k == K_max-1 or (k < K_max-1 and D[g][k+1][j][i] == 0):
                            c1, c2 = DIFZCOEF_TB(D[g][k-1][j][i], D[g][k][j][i], dz, BC_top)
                            DZ[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m]-1)] += c2
                            if k > 0:
                                DZ[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m-(I_max*J_max)]-1)] += -c1
                        else:
                            c1, c2, c3 = DIFZCOEF(D[g][k-1][j][i], D[g][k][j][i], D[g][k+1][j][i], dz)
                            if k > 0:
                                DZ[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m-(I_max*J_max)]-1)] += -c1
                            DZ[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m]-1)] += c2
                            if k < K_max-1:
                                DZ[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m+(I_max*J_max)]-1)] += -c3

    matrix = DX + DY + DZ
    print("D_mat generated")
    return matrix

def ADJOINT_TOT_3D_rect_matrix(g, N, conv, TOT):
    max_conv = max(conv)
    matrix = lil_matrix((g*max_conv, g*max_conv))
    for group in range(g):
        for n in range(N):
            matrix[group*max_conv+(conv[n]-1), group*max_conv+(conv[n]-1)] += TOT[group][n]
    print("TOT_mat generated")
    return matrix.transpose()

def ADJOINT_SCAT_3D_rect_matrix(g, N, conv, SIGS):
    max_conv = max(conv)
    matrix = lil_matrix((g*max_conv, g*max_conv))
    if g == 1:
        for i in range(N):
            matrix[(conv[k]-1), (conv[k]-1)] += SIGS[0][i]
    else:
        for i in range(g):
            for j in range(g):
                for k in range(N):
                    matrix[i*max_conv + (conv[k]-1), j*max_conv + (conv[k]-1)] += SIGS[i][j][k]
    print("SCAT_mat generated")
    return matrix.transpose()

def ADJOINT_NUFIS_3D_rect_matrix(g, N, conv, chi, NUFIS):
    max_conv = max(conv)
    matrix = lil_matrix((g*max_conv,g*max_conv))
    for i in range(g):
        for j in range(g):
            for k in range(N):
                    matrix[i*max_conv + (conv[k]-1), j*max_conv + (conv[k]-1)] += chi[i][k]*NUFIS[j][k]
    print("NUFIS_mat generated")
    return matrix.transpose()

##############################################################################
def NOISE_D_3D_rect_matrix(group, BC, conv, dx, dy, dz, D):
    def DIFXCOEF(D_west, D_mid, D_east, dx):
        a1 = (2*D_mid*D_west)/((D_west+D_mid)*dx**2)
        a2 = (2*D_mid*D_west)/((D_west+D_mid)*dx**2) + (2*D_east*D_mid)/((D_east+D_mid)*dx**2)
        a3 = (2*D_east*D_mid)/((D_east+D_mid)*dx**2)
        return a1, a2, a3
    
    def DIFXCOEF_WB(D_mid, D_east, dx, BC_west):
        if BC_west == 1:  # Zero Flux
            a2 = (2*D_mid)/(dx**2) + (2*D_east*D_mid)/((D_east+D_mid)*dx**2)
        elif BC_west == 2:  # Reflective
            a2 = (2*D_east*D_mid)/((D_east+D_mid)*dx**2)
        elif BC_west == 3:  # Vacuum
            a2 = (2*D_mid)/((4*D_mid*dx)+dx**2) + (2*D_east*D_mid)/((D_east+D_mid)*dx**2)

        a3 = (2*D_east*D_mid)/((D_east+D_mid)*dx**2)
        return a2, a3
    
    def DIFXCOEF_EB(D_west, D_mid, dx, BC_east):
        if BC_east == 1:  # Zero Flux
            a2 = (2*D_mid)/(dx**2) + (2*D_west*D_mid)/((D_west+D_mid)*dx**2)
        elif BC_east == 2:  # Reflective
            a2 = (2*D_west*D_mid)/((D_west+D_mid)*dx**2)
        elif BC_east == 3:  # Vacuum
            a2 = (2*D_mid)/((4*D_mid*dx)+dx**2) + (2*D_mid*D_west)/((D_west+D_mid)*dx**2)

        a1 = (2*D_mid*D_west)/((D_west+D_mid)*dx**2)
        return a1, a2

    def DIFYCOEF(D_bot, D_mid, D_top, dy):
        b1 = (2*D_mid*D_bot)/((D_bot+D_mid)*dy**2)
        b2 = (2*D_mid*D_bot)/((D_bot+D_mid)*dy**2) + (2*D_top*D_mid)/((D_top+D_mid)*dy**2)
        b3 = (2*D_top*D_mid)/((D_top+D_mid)*dy**2)
        return b1, b2, b3

    def DIFYCOEF_SB(D_mid, D_top, dy, BC_south):
        if BC_south == 1:  # Zero Flux
            b2 = (2*D_mid)/(dy**2) + (2*D_top*D_mid)/((D_top+D_mid)*dy**2)
        elif BC_south == 2:  # Reflective
            b2 = (2*D_top*D_mid)/((D_top+D_mid)*dy**2)
        elif BC_south == 3:  # Vacuum
            b2 = (2*D_mid)/((4*D_mid*dy)+dy**2) + (2*D_top*D_mid)/((D_top+D_mid)*dy**2)

        b3 = (2*D_top*D_mid)/((D_top+D_mid)*dy**2)
        return b2, b3

    def DIFYCOEF_NB(D_bot, D_mid, dy, BC_north):
        if BC_north == 1:  # Zero Flux
            b2 = (2*D_mid)/(dy**2) + (2*D_mid*D_bot)/((D_bot+D_mid)*dy**2)
        elif BC_north == 2:  # Reflective
            b2 = (2*D_mid*D_bot)/((D_bot+D_mid)*dy**2)
        elif BC_north == 3:  # Vacuum
            b2 = (2*D_mid)/((4*D_mid*dy)+dy**2) + (2*D_mid*D_bot)/((D_bot+D_mid)*dy**2)

        b1 = (2*D_mid*D_bot)/((D_bot+D_mid)*dy**2)
        return b1, b2

    def DIFZCOEF(D_bot, D_mid, D_top, dz):
        c1 = (2*D_mid*D_bot)/((D_bot+D_mid)*dz**2)
        c2 = (2*D_mid*D_bot)/((D_bot+D_mid)*dz**2) + (2*D_top*D_mid)/((D_top+D_mid)*dz**2)
        c3 = (2*D_top*D_mid)/((D_top+D_mid)*dz**2)
        return c1, c2, c3

    def DIFZCOEF_BB(D_mid, D_top, dz, BC_down):
        if BC_down == 1:  # Zero Flux
            c2 = (2*D_mid)/(dz**2) + (2*D_top*D_mid)/((D_top+D_mid)*dz**2)
        elif BC_down == 2:  # Reflective
            c2 = (2*D_top*D_mid)/((D_top+D_mid)*dz**2)
        elif BC_down == 3:  # Vacuum
            c2 = (2*D_mid)/((4*D_mid*dz)+dz**2) + (2*D_top*D_mid)/((D_top+D_mid)*dz**2)

        c3 = (2*D_top*D_mid)/((D_top+D_mid)*dz**2)
        return c2, c3

    def DIFZCOEF_TB(D_bot, D_mid, dz, BC_up):
        if BC_up == 1:  # Zero Flux
            c2 = (2*D_mid)/(dz**2) + (2*D_bot*D_mid)/((D_bot+D_mid)*dz**2)
        elif BC_up == 2:  # Reflective
            c2 = (2*D_bot*D_mid)/((D_bot+D_mid)*dz**2)
        elif BC_up == 3:  # Vacuum
            c2 = (2*D_mid)/((4*D_mid*dz)+dz**2) + (2*D_mid*D_bot)/((D_bot+D_mid)*dz**2)

        c1 = (2*D_mid*D_bot)/((D_bot+D_mid)*dz**2)
        return c1, c2
    
    # Initialize the full matrix with zeros
    BC_north = BC[0]
    BC_south = BC[1]
    BC_east = BC[2]
    BC_west = BC[3]
    BC_top = BC[4]
    BC_bottom = BC[5]
    I_max = len(D[0][0][0])  # N row
    J_max = len(D[0][0])  # N column
    K_max = len(D[0])  # N column
    max_conv = max(conv)
    matrix = lil_matrix((group*max_conv, group*max_conv))

    # Build DX
    DX = lil_matrix((group*max_conv, group*max_conv))
    for g in range(group):
        for k in range(K_max):  # Loop over depth
            for j in range(J_max):  # Loop over column
                for i in range(I_max):  # Loop over row
                    m = (k * J_max * I_max) + (j * I_max) + i
                    if D[g][k][j][i] != 0:
                        if i == 0 or (i > 0 and D[g][k][j][i-1] == 0):
                            a2, a3 = DIFXCOEF_WB(D[g][k][j][i], D[g][k][j][i+1], dx, BC_west)
                            DX[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m]-1)] += -a2
                            if i < I_max-1:
                                DX[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m+1]-1)] += a3
                        elif i == I_max-1 or (i < I_max-1 and D[g][k][j][i+1] == 0):
                            a1, a2 = DIFXCOEF_EB(D[g][k][j][i-1], D[g][k][j][i], dx, BC_east)
                            DX[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m]-1)] += -a2
                            if i > 0:
                                DX[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m-1]-1)] += a1
                        else:
                            a1, a2, a3 = DIFXCOEF(D[g][k][j][i-1], D[g][k][j][i], D[g][k][j][i+1], dx)
                            if i > 0:
                                DX[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m-1]-1)] += a1
                            DX[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m]-1)] += -a2
                            if i < I_max-1:
                                DX[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m+1]-1)] += a3

    # Build DY
    DY = lil_matrix((group*max_conv, group*max_conv))
    for g in range(group):
        for k in range(K_max):  # Loop over depth
            for j in range(J_max):  # Loop over column
                for i in range(I_max):  # Loop over row
                    m = (k * J_max * I_max) + (j * I_max) + i
                    if D[g][k][j][i] != 0:
                        if j == 0 or (j > 0 and D[g][k][j-1][i] == 0):
                            b2, b3 = DIFYCOEF_SB(D[g][k][j][i], D[g][k][j+1][i], dy, BC_south)
                            DY[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m]-1)] += -b2
                            if j < J_max-1:
                                DY[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m+I_max]-1)] += b3
                        elif j == J_max-1 or (j < J_max-1 and D[g][k][j+1][i] == 0):
                            b1, b2 = DIFYCOEF_NB(D[g][k][j-1][i], D[g][k][j][i], dy, BC_north)
                            DY[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m]-1)] += -b2
                            if j > 0:
                                DY[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m-I_max]-1)] += b1
                        else:
                            b1, b2, b3 = DIFYCOEF(D[g][k][j-1][i], D[g][k][j][i], D[g][k][j+1][i], dy)
                            if j > 0:
                                DY[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m-I_max]-1)] += b1
                            DY[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m]-1)] += -b2
                            if j < J_max-1:
                                DY[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m+I_max]-1)] += b3

    # Build DZ
    DZ = lil_matrix((group*max_conv, group*max_conv))
    for g in range(group):
        for k in range(K_max):  # Loop over depth
            for j in range(J_max):  # Loop over column
                for i in range(I_max):  # Loop over row
                    m = (k * J_max * I_max) + (j * I_max) + i
                    if D[g][k][j][i] != 0:
                        if k == 0 or (k > 0 and D[g][k-1][j][i] == 0):
                            c2, c3 = DIFZCOEF_BB(D[g][k][j][i], D[g][k+1][j][i], dz, BC_bottom)
                            DZ[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m]-1)] += -c2
                            if k < K_max-1:
                                DZ[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m+(I_max*J_max)]-1)] += c3
                        elif k == K_max-1 or (k < K_max-1 and D[g][k+1][j][i] == 0):
                            c1, c2 = DIFZCOEF_TB(D[g][k-1][j][i], D[g][k][j][i], dz, BC_top)
                            DZ[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m]-1)] += -c2
                            if k > 0:
                                DZ[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m-(I_max*J_max)]-1)] += c1
                        else:
                            c1, c2, c3 = DIFZCOEF(D[g][k-1][j][i], D[g][k][j][i], D[g][k+1][j][i], dz)
                            if k > 0:
                                DZ[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m-(I_max*J_max)]-1)] += c1
                            DZ[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m]-1)] += -c2
                            if k < K_max-1:
                                DZ[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m+(I_max*J_max)]-1)] += c3

    matrix = DX + DY + DZ
    print("D_mat generated")
    return matrix

def NOISE_TOT_3D_rect_matrix(g, N, conv, TOT):
    max_conv = max(conv)
    matrix = lil_matrix((g*max_conv, g*max_conv))
    for group in range(g):
        for n in range(N):
            matrix[group*max_conv+(conv[n]-1), group*max_conv+(conv[n]-1)] += TOT[group][n]
    print("TOT_mat generated")
    return matrix

def NOISE_SCAT_3D_rect_matrix(g, N, conv, SIGS):
    max_conv = max(conv)
    matrix = lil_matrix((g*max_conv, g*max_conv))
    if g == 1:
        for i in range(N):
            matrix[(conv[k]-1), (conv[k]-1)] += SIGS[0][i]
    else:
        for i in range(g):
            for j in range(g):
                for k in range(N):
                    matrix[i*max_conv + (conv[k]-1), j*max_conv + (conv[k]-1)] += SIGS[i][j][k]
    print("SCAT_mat generated")
    return matrix

def NOISE_NUFIS_3D_rect_matrix(g, N, conv, chi_p, chi_d, NUFIS, k_complex, Beff, keff):
    max_conv = max(conv)
    matrix = lil_matrix((g*max_conv,g*max_conv), dtype=complex)
    for i in range(g):
        for j in range(g):
            for k in range(N):
                matrix[i*max_conv + (conv[k]-1), j*max_conv + (conv[k]-1)] += (chi_p[i][k] * (1-Beff)/keff + chi_p[i][k] * k_complex) * NUFIS[j][k]
    print("NUFIS_mat generated")
    return matrix

def NOISE_FREQ_3D_rect_matrix(g, N, conv, omega, v):
    max_conv = max(conv)
    matrix = lil_matrix((g*max_conv, g*max_conv), dtype=complex)
    for group in range(g):
        for n in range(N):
            matrix[group*max_conv+(conv[n]-1), group*max_conv+(conv[n]-1)] += 1j*omega/v[group][n]
    print("FREQ_mat generated")
    return matrix

def NOISE_dTOT_3D_rect_matrix(g, N, conv, dTOT):
    max_conv = max(conv)
    matrix = lil_matrix((g*max_conv, g*max_conv), dtype=complex)
    for group in range(g):
        for n in range(N):
            matrix[group*max_conv+(conv[n]-1), group*max_conv+(conv[n]-1)] += dTOT[group][n]
    print("dTOT_mat generated")
    return matrix

def NOISE_dSCAT_3D_rect_matrix(g, N, conv, dSIGS):
    max_conv = max(conv)
    matrix = lil_matrix((g*max_conv, g*max_conv), dtype=complex)
    if g == 1:
        for i in range(N):
            matrix[(conv[k]-1), (conv[k]-1)] += dSIGS[0][i]
    else:
        for i in range(g):
            for j in range(g):
                for k in range(N):
                    matrix[i*max_conv + (conv[k]-1), j*max_conv + (conv[k]-1)] += dSIGS[i][j][k]
    print("dSCAT_mat generated")
    return matrix

def NOISE_dNUFIS_3D_rect_matrix(g, N, conv, chi_p, chi_d, dNUFIS, k_complex, Beff, keff):
    max_conv = max(conv)
    matrix = lil_matrix((g*max_conv,g*max_conv), dtype=complex)
    for i in range(g):
        for j in range(g):
            for k in range(N):
                matrix[i*max_conv + (conv[k]-1), j*max_conv + (conv[k]-1)] += (chi_p[i][k] * (1-Beff)/keff + chi_p[i][k] * k_complex) * dNUFIS[j][k]
    print("dNUFIS_mat generated")
    return matrix

##############################################################################
def plot_heatmap_3D(data, g, z, x, y, cmap='viridis', varname=None, title=None, output_dir=None, case_name=None, process_data=None, solve=None):
    plt.clf()
    if process_data == 'magnitude':
        data = np.abs(data)  # Compute magnitude
    elif process_data == 'phase':
        data_rad = np.angle(data)  # Compute phase
        data = np.degrees(data_rad)  # Compute phase

    extent = [x.min(), x.max(), y.min(), y.max()]
    plt.imshow(data, cmap=cmap, interpolation='nearest', extent=extent, origin='lower')

    if process_data == 'magnitude':
        plt.colorbar(label=f'{varname}{g}')  # Add color bar to show scale
    elif process_data == 'phase':
        plt.colorbar(label=f'{varname}{g}_deg')  # Add color bar to show scale

    if title:
        plt.title(title)
    plt.xlabel('X (cm)')
    plt.ylabel('Y (cm)')
    
    # Define ticks every 10 cm
    x_ticks = np.linspace(x.min(), x.max(), num=10)
    y_ticks = np.linspace(y.min(), y.max(), num=10)
    plt.xticks(x_ticks, labels=[f'{val:.1f}' for val in x_ticks])
    plt.yticks(y_ticks, labels=[f'{val:.1f}' for val in y_ticks])

    filename = f'{output_dir}/{case_name}_{solve}/{case_name}_{solve}_{varname}_{process_data}_G{g}_Z{z}.png'
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.05)
    plt.close()

    return filename

def plot_heatmap_3D_general(data, g, z, x, y, cmap='viridis', title=None, output=None, varname=None, case_name=None, process_data=None):
    plt.clf()
    if process_data == 'magnitude':
        data = np.abs(data)  # Compute magnitude
    elif process_data == 'phase':
        data_rad = np.angle(data)  # Compute phase
        data = np.degrees(data_rad)  # Compute phase

    extent = [x.min(), x.max(), y.min(), y.max()]
    plt.imshow(data, cmap=cmap, interpolation='nearest', extent=extent, origin='lower')

    if process_data == 'magnitude':
        plt.colorbar(label=f'{varname}{g}')  # Add color bar to show scale
    elif process_data == 'phase':
        plt.colorbar(label=f'{varname}{g}_deg')  # Add color bar to show scale

    if title:
        plt.title(title)
    plt.xlabel('X (cm)')
    plt.ylabel('Y (cm)')
    
    # Define ticks every 10 cm
    x_ticks = np.linspace(x.min(), x.max(), num=10)
    y_ticks = np.linspace(y.min(), y.max(), num=10)
    plt.xticks(x_ticks, labels=[f'{val:.1f}' for val in x_ticks])
    plt.yticks(y_ticks, labels=[f'{val:.1f}' for val in y_ticks])

    filename = f'{output}_{varname}_{process_data}_G{g}_Z{z}.png'
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.05)
    plt.close()

    return filename

def plot_heatmap_3D_categorical(data, g, z, x, y, output=None, varname=None, case_name=None, title=None):
    plt.clf()

    # Define categories and corresponding colors
    categories = [0, 0.5, 1, 1.5]
    colors = ['lightgray', 'blue', 'red', 'purple']
    labels = ['Empty', 'Detector', 'Source', 'Detector + Source']

    # Create a discrete colormap
    cmap = mcolors.ListedColormap(colors)
    bounds = [-0.25, 0.25, 0.75, 1.25, 1.75]  # boundaries between categories
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    extent = [x.min(), x.max(), y.min(), y.max()]
    plt.imshow(data, cmap=cmap, norm=norm, interpolation='nearest', extent=extent, origin='lower')

    # Build legend manually (instead of colorbar)
    legend_elements = [Patch(facecolor=colors[i], edgecolor='k', label=labels[i]) for i in range(len(categories))]
    plt.legend(handles=legend_elements,
           loc='upper center',          # center it
           bbox_to_anchor=(0.5, -0.12), # shift below the plot
           ncol=len(categories),        # put all items in one row
           frameon=False)               # remove box around legend

    if title:
        plt.title(title)
    plt.xlabel('X (cm)')
    plt.ylabel('Y (cm)')
    
    # Define ticks every 10 cm
    x_ticks = np.linspace(x.min(), x.max(), num=10)
    y_ticks = np.linspace(y.min(), y.max(), num=10)
    plt.xticks(x_ticks, labels=[f'{val:.1f}' for val in x_ticks])
    plt.yticks(y_ticks, labels=[f'{val:.1f}' for val in y_ticks])

    filename = f'{output}_{varname}_categorical_G{g}_Z{z}.png'
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.05)
    plt.close()

    return filename

##############################################################################
def interpolate_dPHI_rbf_3D_rect(dPHI_zero, group, K_max, J_max, I_max, conv, known_coords, zero_coords, epsilon, rbf_function=None):
    if len(dPHI_zero) == group * max(conv):
        dPHI_zero_new = np.zeros((group * K_max * J_max * I_max), dtype=complex)
        for g in range(group):
            for n in range(K_max * J_max * I_max):
                if conv[n] != 0:
                    dPHI_zero_new[g * (K_max * J_max * I_max) + n] = dPHI_zero[g * max(conv) + (conv[n] - 1)]
    else:
        dPHI_zero_new = dPHI_zero

    dPHI_zero_array = np.reshape(np.array(dPHI_zero_new), (group, K_max, J_max, I_max))
    dPHI_interp_array = dPHI_zero_array.copy()

    for g in range(group):
        dPHI_zero_array_real = np.real(dPHI_zero_array[g])
        dPHI_zero_array_imag = np.imag(dPHI_zero_array[g])

        values_real = np.array([dPHI_zero_array_real[k, j, i] for k, j, i in known_coords])
        values_imag = np.array([dPHI_zero_array_imag[k, j, i] for k, j, i in known_coords])

        # Create RBF interpolator and interpolate for zero elements
        rbf_real = RBFInterpolator(known_coords, values_real, epsilon=epsilon, kernel=rbf_function)
        rbf_imag = RBFInterpolator(known_coords, values_imag, epsilon=epsilon, kernel=rbf_function)

        # Interpolate real and imaginary parts separately only for zero locations
        interpolated_real = rbf_real(zero_coords)
        interpolated_imag = rbf_imag(zero_coords)

        # Handle NaN values in interpolated data by filling with nearest interpolation
        if np.any(np.isnan(interpolated_real)):
            interpolated_real[np.isnan(interpolated_real)] = griddata(
                known_coords, values_real, zero_coords[np.isnan(interpolated_real)], method='linear'
            )
        if np.any(np.isnan(interpolated_imag)):
            interpolated_imag[np.isnan(interpolated_imag)] = griddata(
                known_coords, values_imag, zero_coords[np.isnan(interpolated_imag)], method='linear'
            )

        # Assign interpolated values back to the array
        for idx, (k, j, i) in enumerate(zero_coords):
            dPHI_interp_array[g, k, j, i] = interpolated_real[idx] + 1j * interpolated_imag[idx]

    # Convert the 4D array back to a 1D list
    dPHI_interp = dPHI_interp_array.ravel().tolist()

    # Apply conv-based NaN and zero conditions on dPHI_interp
    for g in range(group):
        start_idx = g * K_max * J_max * I_max
        for n in range(K_max * J_max * I_max):
            global_idx = start_idx + n
            if conv[n] == 0:
                dPHI_interp[global_idx] = np.nan
            elif conv[n] > 0 and np.isnan(dPHI_interp[global_idx]):
                dPHI_interp[global_idx] = 0

    if len(dPHI_zero) == group * max(conv):
        dPHI_interp_new = np.zeros((group * max(conv)), dtype=complex)
        for g in range(group):
            for n in range(K_max * J_max * I_max):
                if conv[n] != 0:
                    dPHI_interp_new[g * max(conv) + (conv[n] - 1)] = dPHI_interp[g * (K_max * J_max * I_max) + n]
    else:
        dPHI_interp_new = dPHI_interp

    return dPHI_interp_new
