import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csc_matrix
import h5py
import os
import sys
from scipy.interpolate import griddata
from scipy.interpolate import RBFInterpolator
from matplotlib.colors import Normalize
from matplotlib import cm
from PIL import Image
from scipy.spatial.distance import pdist, squareform

# Prevent .pyc file generation
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True

#######################################################################################################
# DEFAULT VALUES
output_dir = None
case_name = None
x, y = 0, 0
#######################################################################################################
# Function to convert 2D rectangular indexes
def convert_index_2D_rect(D, I_max, J_max):
    conv = [0] * (I_max*J_max)
    tmp_conv = 0
    for j in range(J_max):  
        for i in range(I_max):
            if D[0][j][i] != 0:
                tmp_conv += 1
                m = j * I_max + i
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
def save_output_hdf5(hdf5_filename, output):
    with h5py.File(hdf5_filename, 'w') as hdf5_file:
        for groupname, data in output.items():
            hdf5_group = hdf5_file.create_group(groupname)
            hdf5_group.create_dataset("real", data=[item["real"] for item in data])
            hdf5_group.create_dataset("imaginary", data=[item["imaginary"] for item in data])

# Function to load data in HDF5 format
def load_output_hdf5(filename):
    output_list = []
    with h5py.File(filename, 'r') as f:
        for key in f.keys():
            real_data = f[f'{key}/real'][:]
            imag_data = f[f'{key}/imaginary'][:]
            complex_data = [complex(real, imag) for real, imag in zip(real_data, imag_data)]
            output_list.extend(complex_data)
    return output_list

##############################################################################
def FORWARD_D_2D_rect_matrix(group, BC, conv, dx, dy, D):
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

    # Initialize the full matrix with zeros
    BC_north = BC[0]
    BC_south = BC[1]
    BC_east = BC[2]
    BC_west = BC[3]
    I_max = len(D[0][0])  # N row
    J_max = len(D[0])  # N column
    max_conv = max(conv)
    matrix = lil_matrix((group*max_conv, group*max_conv))

    # Build DX
    DX = lil_matrix((group*max_conv, group*max_conv))
    for g in range(group):
        for j in range(J_max):  # Loop over column
            for i in range(I_max):  # Loop over row
                m = j * I_max + i
                if D[g][j][i] != 0:
                    if i == 0 or (i > 0 and D[g][j][i-1] == 0):
                        a2, a3 = DIFXCOEF_WB(D[g][j][i], D[g][j][i+1], dx, BC_west)
                        DX[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m]-1)] += a2
                        if i < I_max-1:
                            DX[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m+1]-1)] += -a3
                    elif i == I_max-1 or (i < I_max-1 and D[g][j][i+1] == 0):
                        a1, a2 = DIFXCOEF_EB(D[g][j][i-1], D[g][j][i], dx, BC_east)
                        DX[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m]-1)] += a2
                        if i > 0:
                            DX[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m-1]-1)] += -a1
                    else:
                        a1, a2, a3 = DIFXCOEF(D[g][j][i-1], D[g][j][i], D[g][j][i+1], dx)
                        if i > 0:
                            DX[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m-1]-1)] += -a1
                        DX[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m]-1)] += a2
                        if i < I_max-1:
                            DX[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m+1]-1)] += -a3

    # Build DY
    DY = lil_matrix((group*max_conv, group*max_conv))
    for g in range(group):
        for j in range(J_max):  # Loop over column
            for i in range(I_max):  # Loop over row
                m = j * I_max + i
                if D[g][j][i] != 0:
                    if j == 0 or (j > 0 and D[g][j-1][i] == 0):
                        b2, b3 = DIFYCOEF_SB(D[g][j][i], D[g][j+1][i], dy, BC_south)
                        DY[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m]-1)] += b2
                        if j < J_max-1:
                            DY[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m+I_max]-1)] += -b3
                    elif j == J_max-1 or (j < J_max-1 and D[g][j+1][i] == 0):
                        b1, b2 = DIFYCOEF_NB(D[g][j-1][i], D[g][j][i], dy, BC_north)
                        DY[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m]-1)] += b2
                        if j > 0:
                            DY[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m-I_max]-1)] += -b1
                    else:
                        b1, b2, b3 = DIFYCOEF(D[g][j-1][i], D[g][j][i], D[g][j+1][i], dy)
                        if j > 0:
                            DY[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m-I_max]-1)] += -b1
                        DY[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m]-1)] += b2
                        if j < J_max-1:
                            DY[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m+I_max]-1)] += -b3

    matrix = csc_matrix(DX + DY)  # Convert to CSC format
    print("D_mat generated")

    return matrix

def FORWARD_TOT_2D_rect_matrix(g, N, conv, TOT):
    max_conv = max(conv)
    matrix = lil_matrix((g*max_conv, g*max_conv))
    for group in range(g):
        for n in range(N):
            matrix[group*max_conv+(conv[n]-1), group*max_conv+(conv[n]-1)] += TOT[group][n]
    print("TOT_mat generated")
    return matrix

def FORWARD_SCAT_2D_rect_matrix(g, N, conv, SIGS):
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

def FORWARD_NUFIS_2D_rect_matrix(g, N, conv, chi, NUFIS):
    max_conv = max(conv)
    matrix = lil_matrix((g*max_conv,g*max_conv))
    for i in range(g):
        for j in range(g):
            for k in range(N):
                    matrix[i*max_conv + (conv[k]-1), j*max_conv + (conv[k]-1)] += chi[i][k]*NUFIS[j][k]
    print("NUFIS_mat generated")
    return matrix

##############################################################################
def ADJOINT_D_2D_rect_matrix(group, BC, conv, dx, dy, D):
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

    # Initialize the full matrix with zeros
    BC_north = BC[0]
    BC_south = BC[1]
    BC_east = BC[2]
    BC_west = BC[3]
    I_max = len(D[0][0])  # N row
    J_max = len(D[0])  # N column
    max_conv = max(conv)
    matrix = lil_matrix((group*max_conv, group*max_conv))

    # Build DX
    DX = lil_matrix((group*max_conv, group*max_conv))
    for g in range(group):
        for j in range(J_max):  # Loop over column
            for i in range(I_max):  # Loop over row
                m = j * I_max + i
                if D[g][j][i] != 0:
                    if i == 0 or (i > 0 and D[g][j][i-1] == 0):
                        a2, a3 = DIFXCOEF_WB(D[g][j][i], D[g][j][i+1], dx, BC_west)
                        DX[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m]-1)] += a2
                        if i < I_max-1:
                            DX[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m+1]-1)] += -a3
                    elif i == I_max-1 or (i < I_max-1 and D[g][j][i+1] == 0):
                        a1, a2 = DIFXCOEF_EB(D[g][j][i-1], D[g][j][i], dx, BC_east)
                        DX[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m]-1)] += a2
                        if i > 0:
                            DX[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m-1]-1)] += -a1
                    else:
                        a1, a2, a3 = DIFXCOEF(D[g][j][i-1], D[g][j][i], D[g][j][i+1], dx)
                        if i > 0:
                            DX[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m-1]-1)] += -a1
                        DX[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m]-1)] += a2
                        if i < I_max-1:
                            DX[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m+1]-1)] += -a3

    # Build DY
    DY = lil_matrix((group*max_conv, group*max_conv))
    for g in range(group):
        for j in range(J_max):  # Loop over column
            for i in range(I_max):  # Loop over row
                m = j * I_max + i
                if D[g][j][i] != 0:
                    if j == 0 or (j > 0 and D[g][j-1][i] == 0):
                        b2, b3 = DIFYCOEF_SB(D[g][j][i], D[g][j+1][i], dy, BC_south)
                        DY[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m]-1)] += b2
                        if j < J_max-1:
                            DY[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m+I_max]-1)] += -b3
                    elif j == J_max-1 or (j < J_max-1 and D[g][j+1][i] == 0):
                        b1, b2 = DIFYCOEF_NB(D[g][j-1][i], D[g][j][i], dy, BC_north)
                        DY[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m]-1)] += b2
                        if j > 0:
                            DY[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m-I_max]-1)] += -b1
                    else:
                        b1, b2, b3 = DIFYCOEF(D[g][j-1][i], D[g][j][i], D[g][j+1][i], dy)
                        if j > 0:
                            DY[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m-I_max]-1)] += -b1
                        DY[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m]-1)] += b2
                        if j < J_max-1:
                            DY[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m+I_max]-1)] += -b3

    matrix = csc_matrix(DX + DY)  # Convert to CSC format
    print("D_mat generated")

    return matrix

def ADJOINT_TOT_2D_rect_matrix(g, N, conv, TOT):
    max_conv = max(conv)
    matrix = lil_matrix((g*max_conv, g*max_conv))
    for group in range(g):
        for n in range(N):
            matrix[group*max_conv+(conv[n]-1), group*max_conv+(conv[n]-1)] += TOT[group][n]
    print("TOT_mat generated")
    return matrix.transpose()

def ADJOINT_SCAT_2D_rect_matrix(g, N, conv, SIGS):
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

def ADJOINT_NUFIS_2D_rect_matrix(g, N, conv, chi, NUFIS):
    max_conv = max(conv)
    matrix = lil_matrix((g*max_conv,g*max_conv))
    for i in range(g):
        for j in range(g):
            for k in range(N):
                    matrix[i*max_conv + (conv[k]-1), j*max_conv + (conv[k]-1)] += chi[i][k]*NUFIS[j][k]
    print("NUFIS_mat generated")
    return matrix.transpose()

##############################################################################
def NOISE_D_2D_rect_matrix(group, BC, conv, dx, dy, D):
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

    # Initialize the full matrix with zeros
    BC_north = BC[0]
    BC_south = BC[1]
    BC_east = BC[2]
    BC_west = BC[3]
    I_max = len(D[0][0])  # N row
    J_max = len(D[0])  # N column
    max_conv = max(conv)
    matrix = lil_matrix((group*max_conv, group*max_conv))

    # Build DX
    DX = lil_matrix((group*max_conv, group*max_conv))
    for g in range(group):
        for j in range(J_max):  # Loop over column
            for i in range(I_max):  # Loop over row
                m = j * I_max + i
                if D[g][j][i] != 0:
                    if i == 0 or (i > 0 and D[g][j][i-1] == 0):
                        a2, a3 = DIFXCOEF_WB(D[g][j][i], D[g][j][i+1], dx, BC_west)
                        DX[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m]-1)] += -a2
                        if i < I_max-1:
                            DX[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m+1]-1)] += a3
                    elif i == I_max-1 or (i < I_max-1 and D[g][j][i+1] == 0):
                        a1, a2 = DIFXCOEF_EB(D[g][j][i-1], D[g][j][i], dx, BC_east)
                        DX[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m]-1)] += -a2
                        if i > 0:
                            DX[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m-1]-1)] += a1
                    else:
                        a1, a2, a3 = DIFXCOEF(D[g][j][i-1], D[g][j][i], D[g][j][i+1], dx)
                        if i > 0:
                            DX[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m-1]-1)] += a1
                        DX[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m]-1)] += -a2
                        if i < I_max-1:
                            DX[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m+1]-1)] += a3

    # Build DY
    DY = lil_matrix((group*max_conv, group*max_conv))
    for g in range(group):
        for j in range(J_max):  # Loop over column
            for i in range(I_max):  # Loop over row
                m = j * I_max + i
                if D[g][j][i] != 0:
                    if j == 0 or (j > 0 and D[g][j-1][i] == 0):
                        b2, b3 = DIFYCOEF_SB(D[g][j][i], D[g][j+1][i], dy, BC_south)
                        DY[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m]-1)] += -b2
                        if j < J_max-1:
                            DY[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m+I_max]-1)] += b3
                    elif j == J_max-1 or (j < J_max-1 and D[g][j+1][i] == 0):
                        b1, b2 = DIFYCOEF_NB(D[g][j-1][i], D[g][j][i], dy, BC_north)
                        DY[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m]-1)] += -b2
                        if j > 0:
                            DY[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m-I_max]-1)] += b1
                    else:
                        b1, b2, b3 = DIFYCOEF(D[g][j-1][i], D[g][j][i], D[g][j+1][i], dy)
                        if j > 0:
                            DY[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m-I_max]-1)] += b1
                        DY[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m]-1)] += -b2
                        if j < J_max-1:
                            DY[(g*max_conv)+(conv[m]-1), (g*max_conv)+(conv[m+I_max]-1)] += b3

    matrix = csc_matrix(DX + DY)  # Convert to CSC format
    print("D_mat generated")

    return matrix

def NOISE_TOT_2D_rect_matrix(g, N, conv, TOT):
    max_conv = max(conv)
    matrix = lil_matrix((g*max_conv, g*max_conv))
    for group in range(g):
        for n in range(N):
            matrix[group*max_conv+(conv[n]-1), group*max_conv+(conv[n]-1)] += TOT[group][n]
    print("TOT_mat generated")
    return matrix

def NOISE_SCAT_2D_rect_matrix(g, N, conv, SIGS):
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

def NOISE_NUFIS_2D_rect_matrix(g, N, conv, chi_p, chi_d, NUFIS, k_complex, Beff, keff):
    max_conv = max(conv)
    matrix = lil_matrix((g*max_conv,g*max_conv), dtype=complex)
    for i in range(g):
        for j in range(g):
            for k in range(N):
                matrix[i*max_conv + (conv[k]-1), j*max_conv + (conv[k]-1)] += (chi_p[i][k] * (1-Beff)/keff + chi_d[i][k] * k_complex) * NUFIS[j][k]
    print("NUFIS_mat generated")
    return matrix

def NOISE_FREQ_2D_rect_matrix(g, N, conv, omega, v):
    max_conv = max(conv)
    matrix = lil_matrix((g*max_conv, g*max_conv), dtype=complex)
    for group in range(g):
        for n in range(N):
            matrix[group*max_conv+(conv[n]-1), group*max_conv+(conv[n]-1)] += 1j*omega/v[group][n]
    print("FREQ_mat generated")
    return matrix

def NOISE_dTOT_2D_rect_matrix(g, N, conv, dTOT):
    max_conv = max(conv)
    matrix = lil_matrix((g*max_conv, g*max_conv), dtype=complex)
    for group in range(g):
        for n in range(N):
            matrix[group*max_conv+(conv[n]-1), group*max_conv+(conv[n]-1)] += dTOT[group][n]
    print("dTOT_mat generated")
    return matrix

def NOISE_dSCAT_2D_rect_matrix(g, N, conv, dSIGS):
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

def NOISE_dNUFIS_2D_rect_matrix(g, N, conv, chi_p, chi_d, dNUFIS, k_complex, Beff, keff):
    max_conv = max(conv)
    matrix = lil_matrix((g*max_conv,g*max_conv), dtype=complex)
    for i in range(g):
        for j in range(g):
            for k in range(N):
                matrix[i*max_conv + (conv[k]-1), j*max_conv + (conv[k]-1)] += (chi_p[i][k] * (1-Beff)/keff + chi_d[i][k] * k_complex) * dNUFIS[j][k]
    print("dNUFIS_mat generated")
    return matrix

##############################################################################
def generate_dPHI_gif(dPHI, f, group, N, I_max, J_max, output_dir, case_name, solve, max_time=5, num_timesteps=501):
    # Define time steps
    time_steps = np.linspace(0, max_time, num_timesteps)

    # Create separate lists for filenames for each group
    group_filenames = [[] for _ in range(group)]

    z_limit_g = []
    z_magnitude = []
    dPHI_array = np.array(dPHI)
    dPHI_array = np.nan_to_num(dPHI_array, nan=0)
    for g in range(group):
        for n in range(N):
            z_magnitude.append(np.abs(dPHI_array[g*N + n]))
    z_magnitude_array = np.array(z_magnitude).reshape(group, N)
    for g in range(group):
        z_limit_g.append(max(z_magnitude_array[g]) * 1.1)

    for i, t in enumerate(time_steps):
        print(f'Plotting dPHI_time for timestep t = {t} s')
        dPHI_time = dPHI.copy()
        
        for g in range(group):
            for n in range(N):
                magnitude = np.abs(dPHI[g*N + n])
                phase = np.angle(dPHI[g*N + n])

                dPHI_time[g*N + n] = magnitude * np.cos(2 * np.pi * f * t + phase)

        dPHI_time_array = np.array(dPHI_time)
        dPHI_time_reshaped = dPHI_time_array.reshape(group, I_max, J_max)

        # Create X, Y coordinates for bars
        x, y = np.meshgrid(np.arange(I_max), np.arange(J_max), indexing='ij')
        x = x.flatten()
        y = y.flatten()

        for g in range(group):
            z_limit = z_limit_g[g] #np.nanmax(np.abs(dPHI_time_reshaped[g])) * 1.1
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')

            # Set bar positions and heights based on the magnitude of dPHI values
            z = np.zeros_like(x)
            dx = dy = 0.8  # bar width
            dz = np.abs(dPHI_time_reshaped[g, :, :]).flatten()  # height as magnitude of complex number

            # Normalize the colors based on bar height for color mapping
            norm = Normalize(vmin=0, vmax=z_limit)
            colors = cm.viridis(norm(dz))  # Apply colormap (viridis can be changed to any colormap)

            # Plot bars with color
            ax.bar3d(x, y, z, dx, dy, dz, color=colors, shade=True)

            # Add color bar
            mappable = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
            mappable.set_array(dz)
            cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10)
            cbar.set_label('|dPHI|')

            # Setting a consistent limit for the z-axis
            ax.set_zlim(0, z_limit)

            # Labeling
            ax.set_title(f'dPHI in Time Domain for Group {g + 1}\nt = {t:.2f} s')
            ax.set_xlabel('X index')
            ax.set_ylabel('Y index')
            ax.set_zlabel('|dPHI|')
            plt.tight_layout()

            # Save frame for each group
            frame_filename = f'{output_dir}/{case_name}_{solve}/{case_name}_{solve}_time_G{g+1}_t{i:03}.png'
            plt.savefig(frame_filename, bbox_inches='tight', pad_inches=0.05)
            group_filenames[g].append(frame_filename)
            plt.close(fig)  # Close figure to save memory

    # Generate a GIF for each group
    for g, filenames in enumerate(group_filenames):
        print(f'Making GIF for dPHI_time group {g+1}')
        frames = [Image.open(filename) for filename in filenames]
        frames[0].save(
            f'{output_dir}/{case_name}_{solve}/{case_name}_{solve}_time_G{g+1}_animation.gif',
            save_all=True,
            append_images=frames[1:],  # Append remaining frames
            duration=300,               # Duration for each frame in ms
            loop=0                      # Loop forever
        )

        # Optional: Cleanup (delete individual frames if not needed anymore)
        for filename in filenames:
            os.remove(filename)

def plot_heatmap(data, g, x, y, cmap='viridis', varname=None, title=None, output_dir=None, case_name=None, process_data=None, solve=None):
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

    filename = f'{output_dir}/{case_name}_{solve}/{case_name}_{solve}_{varname}_{process_data}_G{g}.png'
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.05)
    plt.close()

    return filename

##############################################################################
def interpolate_dPHI_rbf_2D_rect(dPHI_zero, group, J_max, I_max, conv, map_detector, rbf_function=None):
    if len(dPHI_zero) == group * max(conv):
        dPHI_zero_new = np.zeros((group* I_max * J_max), dtype=complex)
        for g in range(group):
            for n in range(I_max * J_max):
                if conv[n] != 0:
                    dPHI_zero_new[g * (I_max * J_max) + n] = dPHI_zero[g * max(conv) + (conv[n] - 1)]
    else:
        dPHI_zero_new = dPHI_zero

    dPHI_zero_array = np.reshape(np.array(dPHI_zero_new), (group, J_max, I_max))
    dPHI_interp_array = dPHI_zero_array.copy()

    for g in range(group):
        dPHI_zero_array_real = np.real(dPHI_zero_array[g])
        dPHI_zero_array_imag = np.imag(dPHI_zero_array[g])

        # Get non-zero coordinates and values for real and imaginary parts
        coords_real = np.array([(j, i) for j in range(J_max) for i in range(I_max)
                                if map_detector[j * I_max + i] == 1 ])
        values_real = np.array([dPHI_zero_array_real[j, i] for j, i in coords_real])
        coords_imag = np.array([(j, i) for j in range(J_max) for i in range(I_max)
                                if map_detector[j * I_max + i] == 1 ])
        values_imag = np.array([dPHI_zero_array_imag[j, i] for j, i in coords_imag])
        
        # Calculate the pairwise distances between points to determine epsilon
        pairwise_distances = pdist(coords_real, metric='euclidean')
        avg_distance = np.mean(pairwise_distances)  # You can also use np.median or another method
        epsilon = avg_distance / 16  # Set epsilon as a fraction of the average distance

        # Create RBF interpolator and interpolate for zero elements
        rbf_real = RBFInterpolator(coords_real, values_real, epsilon=epsilon, kernel=rbf_function)
        rbf_imag = RBFInterpolator(coords_imag, values_imag, epsilon=epsilon, kernel=rbf_function)

        # Interpolate real and imaginary parts separately only for zero locations
        zero_coords = np.array([(j, i) for j in range(J_max) for i in range(I_max) if map_detector[j * I_max + i] == 0])

        interpolated_real = rbf_real(zero_coords)
        interpolated_imag = rbf_imag(zero_coords)

        # Handle NaN values in interpolated data by filling with nearest interpolation
        if np.any(np.isnan(interpolated_real)):
            interpolated_real[np.isnan(interpolated_real)] = griddata(
                coords_real, values_real, zero_coords[np.isnan(interpolated_real)], method='linear'
            )
        if np.any(np.isnan(interpolated_imag)):
            interpolated_imag[np.isnan(interpolated_imag)] = griddata(
                coords_imag, values_imag, zero_coords[np.isnan(interpolated_imag)], method='linear'
            )

        # Assign interpolated values back to the array
        for idx, (j, i) in enumerate(zero_coords):
            dPHI_interp_array[g, j, i] = interpolated_real[idx] + 1j * interpolated_imag[idx]

    # Convert the 3D array back to a 1D list
    dPHI_interp = dPHI_interp_array.ravel().tolist()

    # Apply conv-based NaN and zero conditions on dPHI_interp
    for g in range(group):
        start_idx = g * J_max * I_max
        for n in range(J_max * I_max):
            global_idx = start_idx + n
            if conv[n] == 0:
                dPHI_interp[global_idx] = np.nan
            elif conv[n] > 0 and np.isnan(dPHI_interp[global_idx]):
                dPHI_interp[global_idx] = 0

    if len(dPHI_zero) == group * max(conv):
        dPHI_interp_new = np.zeros((group * max(conv)), dtype=complex)
        for g in range(group):
            for n in range(I_max * J_max):
                if conv[n] != 0:
                    dPHI_interp_new[g * max(conv) + (conv[n] - 1)] = dPHI_interp[g * (I_max * J_max) + n]
    else:
        dPHI_interp_new = dPHI_interp

    return dPHI_interp_new
