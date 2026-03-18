import numpy as np
from scipy.sparse import lil_matrix
import os
import sys
import h5py
from scipy.interpolate import RBFInterpolator, griddata

# Prevent .pyc file generation
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True

# Function to convert 1D indexes
def convert_index_1D(D, N):
    conv = [0] * (N)
    tmp_conv = 0
    for n in range(N):
        if D[0][n] != 0:
            tmp_conv += 1
            conv[n] = tmp_conv
    return conv

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

# Function to save sparse matrix to file
def save_sparse_matrix(A, filename):
    A_coo = A.tocoo()
    I, J, V = A_coo.row, A_coo.col, A_coo.data
    
    with open(filename, 'w') as file:
        for i, j, v in zip(I, J, V):
            file.write(f"{i} {j} {v}\n")
    
    print(f"Sparse matrix saved to {filename}")

##############################################################################
def FORWARD_D_1D_matrix(group, BC, N, dx, D):
    # Initialize the full matrix with zeros
    matrix = lil_matrix((group*N, group*N))

    # Initialize BC
    BC_left = BC[0]
    BC_right = BC[1]
    
    # Create tridiagonal matrix for upper left quadrant with boundary conditions
    for g in range(group):
        for n in range(N):
            if n == 0:
                if BC_left == 1:  # Zero Flux
                    matrix[g*N, g*N] += ((2 * D[g][0]*D[g][1]) / ((dx**2) * (D[g][0]+D[g][1]))) +((2 * D[g][0]) / (dx**2))
                    matrix[g*N, g*N+1] += -((2 * D[g][0]*D[g][1]) / ((dx**2) * (D[g][0]+D[g][1])))
                elif BC_left == 2:  # Reflective
                    matrix[g*N, g*N] += ((2 * D[g][0]*D[g][1]) / ((dx**2) * (D[g][0]+D[g][1])))
                    matrix[g*N, g*N+1] += -((2 * D[g][0]*D[g][1]) / ((dx**2) * (D[g][0]+D[g][1])))
                elif BC_left == 3:  # Vacuum
                    matrix[g*N, g*N] += ((2 * D[g][0]*D[g][1]) / ((dx**2) * (D[g][0]+D[g][1]))) +((2 * D[g][0]) /((4*D[g][0]*dx)+(dx**2)))
                    matrix[g*N, g*N+1] += -((2 * D[g][0]*D[g][1]) / ((dx**2) * (D[g][0]+D[g][1])))
                else:
                    raise ValueError("Invalid BC")
            elif n == N - 1:
                if BC_right == 1:  # Zero Flux
                    matrix[g*N+n, g*N+n - 1] += -((2 * D[g][n - 1]*D[g][n]) / ((dx**2) * (D[g][n - 1]+D[g][n])))
                    matrix[g*N+n, g*N+n] += ((2 * D[g][n]) / (dx**2)) +((2 * D[g][n - 1]*D[g][n]) / ((dx**2) * (D[g][n - 1]+D[g][n])))
                elif BC_right == 2:  # Reflective
                    matrix[g*N+n, g*N+n - 1] += -((2 * D[g][n - 1]*D[g][n]) / ((dx**2) * (D[g][n - 1]+D[g][n])))
                    matrix[g*N+n, g*N+n] += ((2 * D[g][n - 1]*D[g][n]) / ((dx**2) * (D[g][n - 1]+D[g][n])))
                elif BC_right == 3:  # Vacuum
                    matrix[g*N+n, g*N+n - 1] += -((2 * D[g][n - 1]*D[g][n]) / ((dx**2) * (D[g][n - 1]+D[g][n])))
                    matrix[g*N+n, g*N+n] += ((2 * D[g][n]) /((4*D[g][n]*dx)+(dx**2))) +((2 * D[g][n - 1]*D[g][n]) / ((dx**2) * (D[g][n - 1]+D[g][n])))
                else:
                    raise ValueError("Invalid BC")
            else:
                matrix[g*N+n, g*N+n - 1] += -((2 * D[g][n - 1]*D[g][n]) / ((dx**2) * (D[g][n - 1]+D[g][n])))
                matrix[g*N+n, g*N+n] += (((2 * D[g][n+1]*D[g][n] / ((dx**2) * (D[g][n + 1]+D[g][n]))) + (2 * D[g][n - 1]*D[g][n]) / ((dx**2) * (D[g][n - 1]+D[g][n]))))
                matrix[g*N+n, g*N+n + 1] += -((2 * D[g][n + 1]*D[g][n]) / ((dx**2) * (D[g][n + 1]+D[g][n])))

    return matrix

def FORWARD_NUFIS_1D_matrix(group, N, chi, NUFIS):
    # Initialize the full matrix with zeros
    matrix = lil_matrix((group*N,group*N))
    for i in range(group):
        for j in range(group):
            for k in range(N):
                matrix[i*N + k, j*N + k] = chi[i][k]*NUFIS[j][k]

    return matrix

def FORWARD_SCAT_1D_matrix(group, N, SIGS):
    # Initialize the full matrix with zeros
    matrix = lil_matrix((group*N, group*N))

    if group == 1:
        for i in range(N):
            matrix[i, i] = SIGS[0][i]
    else:
        for i in range(group):
            for j in range(group):
                for k in range(N):
                    matrix[i * N + k, j * N + k] += SIGS[i][j][k]

    return matrix

def FORWARD_TOT_1D_matrix(group, N, TOT):
    # Initialize the full matrix with zeros
    matrix = lil_matrix((group*N, group*N))
    
    # Create tridiagonal matrix for upper left quadrant with boundary conditions
    for g in range(group):
        for n in range(N):
            matrix[g*N+n, g*N+n] += TOT[g][n]

    return matrix

##############################################################################
def ADJOINT_D_1D_matrix(group, BC, N, dx, D):
    # Initialize the full matrix with zeros
    matrix = lil_matrix((group*N, group*N))

    # Initialize BC
    BC_left = BC[0]
    BC_right = BC[1]
    
    # Create tridiagonal matrix for upper left quadrant with boundary conditions
    for g in range(group):
        for n in range(N):
            if n == 0:
                if BC_left == 1:  # Zero Flux
                    matrix[g*N, g*N] += ((2 * D[g][0]*D[g][1]) / ((dx**2) * (D[g][0]+D[g][1]))) +((2 * D[g][0]) / (dx**2))
                    matrix[g*N, g*N+1] += -((2 * D[g][0]*D[g][1]) / ((dx**2) * (D[g][0]+D[g][1])))
                elif BC_left == 2:  # Reflective
                    matrix[g*N, g*N] += ((2 * D[g][0]*D[g][1]) / ((dx**2) * (D[g][0]+D[g][1])))
                    matrix[g*N, g*N+1] += -((2 * D[g][0]*D[g][1]) / ((dx**2) * (D[g][0]+D[g][1])))
                elif BC_left == 3:  # Vacuum
                    matrix[g*N, g*N] += ((2 * D[g][0]*D[g][1]) / ((dx**2) * (D[g][0]+D[g][1]))) +((2 * D[g][0]) /((4*D[g][0]*dx)+(dx**2)))
                    matrix[g*N, g*N+1] += -((2 * D[g][0]*D[g][1]) / ((dx**2) * (D[g][0]+D[g][1])))
                else:
                    raise ValueError("Invalid BC")
            elif n == N - 1:
                if BC_right == 1:  # Zero Flux
                    matrix[g*N+n, g*N+n - 1] += -((2 * D[g][n - 1]*D[g][n]) / ((dx**2) * (D[g][n - 1]+D[g][n])))
                    matrix[g*N+n, g*N+n] += ((2 * D[g][n]) / (dx**2)) +((2 * D[g][n - 1]*D[g][n]) / ((dx**2) * (D[g][n - 1]+D[g][n])))
                elif BC_right == 2:  # Reflective
                    matrix[g*N+n, g*N+n - 1] += -((2 * D[g][n - 1]*D[g][n]) / ((dx**2) * (D[g][n - 1]+D[g][n])))
                    matrix[g*N+n, g*N+n] += ((2 * D[g][n - 1]*D[g][n]) / ((dx**2) * (D[g][n - 1]+D[g][n])))
                elif BC_right == 3:  # Vacuum
                    matrix[g*N+n, g*N+n - 1] += -((2 * D[g][n - 1]*D[g][n]) / ((dx**2) * (D[g][n - 1]+D[g][n])))
                    matrix[g*N+n, g*N+n] += ((2 * D[g][n]) /((4*D[g][n]*dx)+(dx**2))) +((2 * D[g][n - 1]*D[g][n]) / ((dx**2) * (D[g][n - 1]+D[g][n])))
                else:
                    raise ValueError("Invalid BC")
            else:
                matrix[g*N+n, g*N+n - 1] += -((2 * D[g][n - 1]*D[g][n]) / ((dx**2) * (D[g][n - 1]+D[g][n])))
                matrix[g*N+n, g*N+n] += (((2 * D[g][n+1]*D[g][n] / ((dx**2) * (D[g][n + 1]+D[g][n]))) + (2 * D[g][n - 1]*D[g][n]) / ((dx**2) * (D[g][n - 1]+D[g][n]))))
                matrix[g*N+n, g*N+n + 1] += -((2 * D[g][n + 1]*D[g][n]) / ((dx**2) * (D[g][n + 1]+D[g][n])))

    return matrix

def ADJOINT_TOT_1D_matrix(group, N, TOT):
    # Initialize the full matrix with zeros
    matrix = lil_matrix((group*N, group*N))
    
    # Create tridiagonal matrix for upper left quadrant with boundary conditions
    for g in range(group):
        for n in range(N):
            matrix[g*N+n, g*N+n] += TOT[g][n]

    return matrix.transpose()

def ADJOINT_SCAT_1D_matrix(group, N, SIGS):
    # Initialize the full matrix with zeros
    matrix = lil_matrix((group*N, group*N))

    if group == 1:
        for i in range(N):
            matrix[i, i] = SIGS[0][i]
    else:
        for i in range(group):
            for j in range(group):
                for k in range(N):
                    matrix[i * N + k, j * N + k] += SIGS[i][j][k]

    return matrix.transpose()

def ADJOINT_NUFIS_1D_matrix(group, N, chi, NUFIS):
    # Initialize the full matrix with zeros
    matrix = lil_matrix((group*N,group*N))
    for i in range(group):
        for j in range(group):
            for k in range(N):
                matrix[i*N + k, j*N + k] = chi[i][k]*NUFIS[j][k]

    return matrix.transpose()

##############################################################################
def NOISE_D_1D_matrix(group, BC, N, dx, D):
    # Initialize the full matrix with zeros
    matrix = lil_matrix((group*N, group*N))

    # Initialize BC
    BC_left = BC[0]
    BC_right = BC[1]
    
    # Create tridiagonal matrix for upper left quadrant with boundary conditions
    for g in range(group):
        for n in range(N):
            if n == 0:
                if BC_left == 1:  # Zero Flux
                    matrix[g*N, g*N] += -((2 * D[g][0]*D[g][1]) / ((dx**2) * (D[g][0]+D[g][1]))) -((2 * D[g][0]) / (dx**2))
                    matrix[g*N, g*N+1] += ((2 * D[g][0]*D[g][1]) / ((dx**2) * (D[g][0]+D[g][1])))
                elif BC_left == 2:  # Reflective
                    matrix[g*N, g*N] += -((2 * D[g][0]*D[g][1]) / ((dx**2) * (D[g][0]+D[g][1])))
                    matrix[g*N, g*N+1] += ((2 * D[g][0]*D[g][1]) / ((dx**2) * (D[g][0]+D[g][1])))
                elif BC_left == 3:  # Vacuum
                    matrix[g*N, g*N] += -((2 * D[g][0]*D[g][1]) / ((dx**2) * (D[g][0]+D[g][1]))) -((2 * D[g][0]) /((4*D[g][0]*dx)+(dx**2)))
                    matrix[g*N, g*N+1] += ((2 * D[g][0]*D[g][1]) / ((dx**2) * (D[g][0]+D[g][1])))
                else:
                    raise ValueError("Invalid BC")
            elif n == N - 1:
                if BC_right == 1:  # Zero Flux
                    matrix[g*N+n, g*N+n - 1] += ((2 * D[g][n - 1]*D[g][n]) / ((dx**2) * (D[g][n - 1]+D[g][n])))
                    matrix[g*N+n, g*N+n] += -((2 * D[g][n]) / (dx**2)) -((2 * D[g][n - 1]*D[g][n]) / ((dx**2) * (D[g][n - 1]+D[g][n])))
                elif BC_right == 2:  # Reflective
                    matrix[g*N+n, g*N+n - 1] += ((2 * D[g][n - 1]*D[g][n]) / ((dx**2) * (D[g][n - 1]+D[g][n])))
                    matrix[g*N+n, g*N+n] += -((2 * D[g][n - 1]*D[g][n]) / ((dx**2) * (D[g][n - 1]+D[g][n])))
                elif BC_right == 3:  # Vacuum
                    matrix[g*N+n, g*N+n - 1] += ((2 * D[g][n - 1]*D[g][n]) / ((dx**2) * (D[g][n - 1]+D[g][n])))
                    matrix[g*N+n, g*N+n] += -((2 * D[g][n]) /((4*D[g][n]*dx)+(dx**2))) -((2 * D[g][n - 1]*D[g][n]) / ((dx**2) * (D[g][n - 1]+D[g][n])))
                else:
                    raise ValueError("Invalid BC")
            else:
                matrix[g*N+n, g*N+n - 1] += ((2 * D[g][n - 1]*D[g][n]) / ((dx**2) * (D[g][n - 1]+D[g][n])))
                matrix[g*N+n, g*N+n] += -(((2 * D[g][n+1]*D[g][n] / ((dx**2) * (D[g][n + 1]+D[g][n]))) + (2 * D[g][n - 1]*D[g][n]) / ((dx**2) * (D[g][n - 1]+D[g][n]))))
                matrix[g*N+n, g*N+n + 1] += ((2 * D[g][n + 1]*D[g][n]) / ((dx**2) * (D[g][n + 1]+D[g][n])))

    return matrix

def NOISE_TOT_1D_matrix(group, N, TOT):
    # Initialize the full matrix with zeros
    matrix = lil_matrix((group*N, group*N), dtype=complex)
    
    # Create tridiagonal matrix for upper left quadrant with boundary conditions
    for g in range(group):
        for n in range(N):
            matrix[g*N+n, g*N+n] += TOT[g][n]

    return matrix

def NOISE_SCAT_1D_matrix(group, N, SIGS):
    # Initialize the full matrix with zeros
    matrix = lil_matrix((group*N, group*N), dtype=complex)

    if group == 1:
        for i in range(N):
            matrix[i, i] = SIGS[0][i]
    else:
        for i in range(group):
            for j in range(group):
                for k in range(N):
                    matrix[i * N + k, j * N + k] += SIGS[i][j][k]

    return matrix

def NOISE_NUFIS_1D_matrix(group, N, chi_p, chi_d, NUFIS, k_complex, Beff, keff):
    # Initialize the full matrix with zeros
    matrix = lil_matrix((group*N,group*N), dtype=complex)
    for i in range(group):
        for j in range(group):
            for k in range(N):
                matrix[i*N + k, j*N + k] = (chi_p[i][k] * (1-Beff)/keff + chi_d[i][k] * k_complex) * NUFIS[j][k]

    return matrix

def NOISE_FREQ_1D_matrix(group, N, omega, v):
    # Initialize the full matrix with zeros
    matrix = lil_matrix((group*N, group*N), dtype=complex)
    
    # Create tridiagonal matrix for upper left quadrant with boundary conditions
    for g in range(group):
        for n in range(N):
            matrix[g*N+n, g*N+n] += 1j*omega/v[g][n]

    return matrix

def NOISE_dTOT_1D_matrix(group, N, dTOT):
    # Initialize the full matrix with zeros
    matrix = lil_matrix((group*N, group*N), dtype=complex)
    
    # Create tridiagonal matrix for upper left quadrant with boundary conditions
    for g in range(group):
        for n in range(N):
            matrix[g*N+n, g*N+n] += dTOT[g][n]

    return matrix

def NOISE_dSCAT_1D_matrix(group, N, dSIGS):
    # Initialize the full matrix with zeros
    matrix = lil_matrix((group*N, group*N), dtype=complex)

    if group == 1:
        for i in range(N):
            matrix[i, i] = dSIGS[0][i]
    else:
        for i in range(group):
            for j in range(group):
                for k in range(N):
                    matrix[i * N + k, j * N + k] += dSIGS[i][j][k]

    return matrix

def NOISE_dNUFIS_1D_matrix(group, N, chi_p, chi_d, dNUFIS, k_complex, Beff, keff):
    # Initialize the full matrix with zeros
    matrix = lil_matrix((group*N,group*N), dtype=complex)
    for i in range(group):
        for j in range(group):
            for k in range(N):
                matrix[i*N + k, j*N + k] += (chi_p[i][k] * (1-Beff)/keff + chi_d[i][k] * k_complex) * dNUFIS[j][k]

    return matrix

###################################################################################################
def interpolate_dPHI_rbf_1D(dPHI_zero, group, N, map_detector, rbf_function=None):
    """
    Interpolate dPHI values in 1D using RBF interpolation.

    Parameters:
        dPHI_zero (array): Input complex array with zeroed elements.
        group (int): Number of energy groups.
        N_max (int): Total number of elements in the 1D grid.
        conv (array): Conversion array mapping indices.
        map_detector (array): Binary array indicating detector positions.
        rbf_function (str): RBF kernel function (e.g., 'multiquadric', 'gaussian').

    Returns:
        dPHI_interp_new (array): Interpolated complex array.
    """
    # Reshape the input into a 2D array of shape (group, N_max)
    dPHI_zero_array = np.reshape(np.array(dPHI_zero), (group, N))
    dPHI_interp_array = dPHI_zero_array.copy()

    for g in range(group):
        dPHI_zero_real = np.real(dPHI_zero_array[g])
        dPHI_zero_imag = np.imag(dPHI_zero_array[g])

        # Get non-zero coordinates and values for real and imaginary parts
        coords_real = np.array([n for n in range(N) if map_detector[n] == 1]).reshape(-1, 1)
        values_real = np.array([dPHI_zero_real[n] for n in coords_real.flatten()])
        coords_imag = coords_real  # Same coordinates for real and imaginary parts
        values_imag = np.array([dPHI_zero_imag[n] for n in coords_imag.flatten()])

        # Calculate epsilon based on the pairwise distances
        pairwise_distances = np.diff(coords_real.flatten())
        avg_distance = np.mean(pairwise_distances)
        epsilon = avg_distance / (32 * np.max(values_real))
        
        # Create RBF interpolators
        rbf_real = RBFInterpolator(coords_real, values_real, epsilon=epsilon, kernel=rbf_function)
        rbf_imag = RBFInterpolator(coords_imag, values_imag, epsilon=epsilon, kernel=rbf_function)

        # Interpolate for zero elements
        zero_coords = np.array([n for n in range(N) if map_detector[n] == 0]).reshape(-1, 1)
        interpolated_real = rbf_real(zero_coords)
        interpolated_imag = rbf_imag(zero_coords)

        # Handle NaN values using nearest interpolation
        if np.any(np.isnan(interpolated_real)):
            interpolated_real[np.isnan(interpolated_real)] = griddata(
                coords_real.flatten(), values_real, zero_coords[np.isnan(interpolated_real)], method='nearest'
            )
        if np.any(np.isnan(interpolated_imag)):
            interpolated_imag[np.isnan(interpolated_imag)] = griddata(
                coords_imag.flatten(), values_imag, zero_coords[np.isnan(interpolated_imag)], method='nearest'
            )

        # Assign interpolated values back to the array
        for idx, n in enumerate(zero_coords.flatten()):
            dPHI_interp_array[g, n] = interpolated_real[idx] + 1j * interpolated_imag[idx]

    # Flatten the array back to list
    dPHI_interp = dPHI_interp_array.tolist()

    # Convert back to compact representation if necessary
    if len(dPHI_zero) == group * N:
        dPHI_interp_new = np.zeros((group * N), dtype=complex)
        for g in range(group):
            for n in range(N):
                dPHI_interp_new[g * N + n] = dPHI_interp[g][n]
    else:
        dPHI_interp_new = dPHI_interp

    return dPHI_interp_new