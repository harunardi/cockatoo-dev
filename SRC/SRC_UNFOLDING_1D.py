import numpy as np
import json
import time
import os
import sys
from scipy.linalg import lstsq
import scipy.linalg
from itertools import combinations, islice
from math import comb
from scipy.linalg import lu_factor, lu_solve

# Prevent .pyc file generation
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True

start_time = time.time()

from SRC.UTILS import Utils
from SRC.MATRIX_BUILDER import *
from SRC.METHODS import *
from SRC.POSTPROCESS import PostProcessor
from SRC.SOLVERFACTORY import SolverFactory

#######################################################################################################
def main_unfold_1D_noise(PHI, keff, group, N, TOT, SIGS, BC, dx, D, chi, NUFIS, precond, v, Beff, omega, l, dTOT, dSIGS, dNUFIS, dSOURCE, output_dir, case_name, x):

##### Noise Simulation
    solver_type = 'noise'
    os.makedirs(f'{output_dir}/{case_name}_{solver_type.upper()}', exist_ok=True)

    matrix_builder = MatrixBuilderNoise1D(group, N, TOT, SIGS, BC, dx, D, chi, NUFIS, keff, v, Beff, omega, l, dTOT, dSIGS, dNUFIS)
    M, dS = matrix_builder.build_noise_matrices()
    solver = SolverFactory.get_solver_fixed1D(solver_type, group, N, M, dS, dSOURCE, PHI, precond, tol=1e-06)
    dPHI = solver.solve()
    dPHI_reshaped = np.reshape(dPHI, (group, N))
    output = {}
    for g in range(len(dPHI_reshaped)):
        dPHI_groupname = f'dPHI{g + 1}'
        dPHI_list = [{"real": x.real, "imaginary": x.imag} for x in dPHI_reshaped[g]]
        output[dPHI_groupname] = dPHI_list
    with open(f'{output_dir}/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_output.json', 'w') as json_file:
        json.dump(output, json_file, indent=4)

    return dPHI

def main_unfold_1D_green(PHI, keff, group, N, TOT, SIGS, BC, dx, D, chi, NUFIS, precond, v, Beff, omega, l, dTOT, dSIGS, dNUFIS, dSOURCE, output_dir, case_name, x):

    solver_type = 'noise'

##### 01. Green's Function Generation
    os.makedirs(f'{output_dir}/{case_name}_01_GENERATE', exist_ok=True)
    matrix_builder = MatrixBuilderNoise1D(group, N, TOT, SIGS, BC, dx, D, chi, NUFIS, keff, v, Beff, omega, l, dTOT, dSIGS, dNUFIS)
    M, dS = matrix_builder.build_noise_matrices()
    G_sol = np.ones(group*N, dtype=complex)
    G_matrix = np.zeros((group * N, group * N), dtype=complex)
    if precond == 1:
        print('Solving using ILU')
        M_csc = M.tocsc()
        ilu = spilu(M_csc)
        M_preconditioner = LinearOperator(M_csc.shape, matvec=ilu.solve)
    elif precond == 2:
        print('Solving using LU Decomposition')
        M_csc = M.tocsc()
        lu = splu(M_csc)
        M_preconditioner = LinearOperator(M_csc.shape, matvec=lu.solve)
    else:
        print('Solving using Solver')
    for g in range(group):
        for n in range(N):
            dS = [0] * (group * N)
            m = g*N+n
            dS[m] = 1
            errdPHI = 1
            tol = 1E-8
            iter = 0
            while errdPHI > tol:
                G_solold = np.copy(G_sol)
                if precond == 0:
                    G_sol = spsolve(M, dS)
                elif precond == 1:
                    # Solve the linear system with CG and ILU preconditioning
                    G_sol, info = cg(M, dS, tol=1e-8, maxiter=1000, M=M_preconditioner)
                errdPHI = np.max(np.abs(G_sol - G_solold) / (np.abs(G_sol) + 1E-20))
            G_sol_reshape = np.reshape(G_sol, (group, N))
            G_matrix[:, m] = G_sol.flatten()  # Assign solution to row
            # OUTPUT
            output = {}
            for j in range(group):
                G_sol_groupname = f'G{g+1}{j+1}'
                G_sol_list = [{"real": x.real, "imaginary": x.imag} for x in G_sol_reshape[j]]
                output[G_sol_groupname] = G_sol_list
            # Save data to JSON file
            with open(f'{output_dir}/{case_name}_01_GENERATE/Green_g{g+1}_n{n+1}.json', 'w') as json_file:
                json.dump(output, json_file, indent=4)
            print(f'Generated Green Function for group = {g+1}, N = {n+1}')

    return G_matrix

def main_unfold_1D_solve(PHI, G_matrix, dPHI, keff, group, N, TOT, SIGS, BC, dx, D, chi, NUFIS, precond, v, Beff, omega, l, dTOT, dSIGS, dNUFIS, dSOURCE, map_detector, output_dir, case_name, x):

    solver_type = 'noise'
    dPHI_reshaped = np.reshape(dPHI, (group, N))

##### 02. Solve
    os.makedirs(f'{output_dir}/{case_name}_02_SOLVE', exist_ok=True)
    matrix_builder = MatrixBuilderNoise1D(group, N, TOT, SIGS, BC, dx, D, chi, NUFIS, keff, v, Beff, omega, l, dTOT, dSIGS, dNUFIS)
    M, dS = matrix_builder.build_noise_matrices()
    dSOURCE_SOLVE = [item for sublist in dSOURCE for item in sublist] if all(isinstance(sublist, list) for sublist in dSOURCE) else dSOURCE
    S = dS.dot(PHI) + dSOURCE_SOLVE

    # Iterate over group pairs to compute dPHI
    dPHI_SOLVE = np.zeros((group * N), dtype=complex)
    for i in range(group):
        for j in range(group):
            # Extract the relevant blocks from G_matrix and S
            G_block = G_matrix[i*N:(i+1)*N, j*N:(j+1)*N]
            S_block = S[j*N:(j+1)*N]
            # Perform the matrix-vector multiplication for the Green's function
            dPHI_SOLVE[i*N:(i+1)*N] += np.dot(G_block, S_block)
    dPHI_SOLVE_reshaped = np.reshape(dPHI_SOLVE,(group,N))

    # UNFOLDING
    S_reshaped = np.reshape(S,(group,N))
    G_inverse = np.linalg.inv(G_matrix)
    dS_unfold_SOLVE = np.dot(G_inverse, dPHI_SOLVE)
    dS_unfold_SOLVE_reshaped = np.reshape(dS_unfold_SOLVE,(group,N))

    plt.figure()
    for g in range(group):
        plt.clf()
        plt.plot(x, np.abs(dPHI_reshaped[g]) / np.max(np.abs(dPHI_reshaped[0])), 'b-', label=f'Group {g+1} - Direct')
        plt.plot(x, np.abs(dPHI_SOLVE_reshaped[g]) / np.max(np.abs(dPHI_SOLVE_reshaped[0])), 'r--', label=f'Group {g+1} - Green')
        plt.xlabel('height (cm)')
        plt.ylabel('magnitude of flux perturbation')
        plt.title('normalized magnitude of flux perturbation')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{output_dir}/{case_name}_02_SOLVE/{case_name}_02_SOLVE_abs_G{g+1}.png')
        plt.clf()

        plt.plot(x, np.angle(dPHI_reshaped[g]), 'b-', label=f'Group {g+1} - Direct')
        plt.plot(x, np.angle(dPHI_SOLVE_reshaped[g]), 'r--', label=f'Group {g+1} - Green')
        plt.xlabel('height (cm)')
        plt.ylabel('phase of flux perturbation')
        plt.title('phase of flux perturbation')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{output_dir}/{case_name}_02_SOLVE/{case_name}_02_SOLVE_phase_G{g+1}.png')
        plt.clf()

        plt.plot(x, dPHI_reshaped[g].real, 'b-', label=f'Group {g+1} - Direct')
        plt.plot(x, dPHI_SOLVE_reshaped[g].real, 'r--', label=f'Group {g+1} - Green')
        plt.xlabel('height (cm)')
        plt.ylabel('Real component of flux perturbation')
        plt.title('Real component of flux perturbation')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{output_dir}/{case_name}_02_SOLVE/{case_name}_02_SOLVE_real_G{g+1}.png')
        plt.clf()

        plt.plot(x, np.abs(S_reshaped[g]), 'b-', label=f'Group {g+1} - dS Input')
        plt.plot(x, np.abs(dS_unfold_SOLVE_reshaped[g]), 'r--', label=f'Group {g+1} - dS Unfold')
        plt.xlabel('height (cm)')
        plt.ylabel('magnitude of dS')
        plt.title('magnitude of noise unfolded')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{output_dir}/{case_name}_02_SOLVE/{case_name}_02_SOLVE_dS_G{g+1}.png')
        plt.close()

    # OUTPUT
    output = {}
    for g in range(group):
        dPHI_groupname = f'dPHI{g+1}'
        dPHI_list = [{"real": x.real, "imaginary": x.imag} for x in dPHI_SOLVE_reshaped[g]]
        output[dPHI_groupname] = dPHI_list
        S_groupname = f'S{g+1}'
        S_list = [{"real": x.real, "imaginary": x.imag} for x in S_reshaped[g]]
        output[S_groupname] = S_list

    # Save data to JSON file
    with open(f'{output_dir}/{case_name}_02_SOLVE/{case_name}_02_SOLVE_output.json', 'w') as json_file:
        json.dump(output, json_file, indent=4)
    print(f'Generating JSON output for dS')
    output = {}
    for g in range(group):
        dS_unfold_groupname = f'dS_unfold{g+1}'
        dS_unfold_list = [{"real": x.real, "imaginary": x.imag} for x in dS_unfold_SOLVE_reshaped[g]]
        output[dS_unfold_groupname] = dS_unfold_list

    # Save data to JSON file
    with open(f'{output_dir}/{case_name}_02_SOLVE/{case_name}_02_SOLVE_dS_output.json', 'w') as json_file:
        json.dump(output, json_file, indent=4)

    # --------------- MANIPULATE dPHI -------------------
    dPHI_meas = dPHI.copy()
    for g in range(group):
        for n in range(len(map_detector)):
            if map_detector[n] == 0:
                idx = g * N + n
                dPHI_meas[idx] = 0

    return S, dPHI_meas

#######################################################################################################
def main_unfold_1D_invert(dPHI_meas, dPHI, S, G_matrix, group, N, map_detector, output_dir, case_name, x):
    dPHI_reshaped = np.reshape(dPHI, (group, N))
    S_reshaped = np.reshape(S, (group, N))

###### 03. INVERT
    os.makedirs(f'{output_dir}/{case_name}_03_INVERT', exist_ok=True)

    # --------------- MANIPULATE dPHI -------------------
    # Plot dPHI_zero_reshaped
    dPHI_meas_reshaped = np.reshape(dPHI_meas, (group, N)) #3D array, size (group, J_max, I_max)

    # --------------- INTERPOLATE dPHI -------------------
    # Create a copy to avoid modifying the original array
    dPHI_interp = interpolate_dPHI_rbf_1D(dPHI_meas, group, N, map_detector, rbf_function='thin_plate_spline')
    for g in range(group):
        for n in range(N):
            if map_detector[n] == 1:
                dPHI_interp[g*N+n] = dPHI_meas[g*N+n]
    dPHI_INVERT = dPHI_interp
    dPHI_interp_reshaped = np.reshape(dPHI_interp, (group, N))

    # Plotting of the neutron noise induced by the noise source #1 in the frequency domain
    for g in range(group):
        plt.figure()
        plt.plot(x, np.abs(dPHI_reshaped[g])/np.max(np.abs(dPHI_reshaped[0])), 'g-', label=f'Group {g+1} - dPHI_sol')
        plt.plot(x, np.abs(dPHI_interp_reshaped[g])/np.max(np.abs(dPHI_reshaped[0])), 'r--', label=f'Group {g+1} - dPHI_interp')
        plt.legend()
        plt.ylabel('Normalized amplitude of the induced neutron noise')
        plt.title(f'Magnitude of neutron noise - Group {g+1}')
        plt.xlabel('Distance from core centre [cm]')
        plt.grid()
        plt.savefig(f'{output_dir}/{case_name}_03_INVERT/{case_name}_dPHI_interp_magnitude_G{g+1}.png')
        plt.close()
        
    for g in range(group):
        plt.figure()
        plt.plot(x, np.degrees(np.angle(dPHI_reshaped[g])), 'g-', label=f'Group {g+1} - dPHI_sol')
        plt.plot(x, np.degrees(np.angle(dPHI_interp_reshaped[g])), 'r--', label=f'Group {g+1} - dPHI_interp')
        plt.legend()
        plt.ylabel('Phase of the induced neutron noise')
        plt.title(f'Phase of neutron noise - Group {g+1}')
        plt.xlabel('Distance from core centre [cm]')
        plt.grid()
        plt.savefig(f'{output_dir}/{case_name}_03_INVERT/{case_name}_dPHI_interp_phase_G{g+1}.png')
        plt.close()

    # --------------- INTERPOLATE GREEN'S FUNCTION -------------------
    # Delete G_matrix_full at unknown position (column-wise at specific row)
    G_matrix_meas = G_matrix.copy()
    for g in range(group):
        for n in range(N):
            if map_detector[n] == 0:
                G_matrix_meas[g * N + n, :] = 0 # Zeroing a column instead of a row

    # Interpolate columns of the Green's function
    G_matrix_interp = G_matrix_meas
    G_matrix_interp_cols = np.zeros((group * N, group * N), dtype=complex) #np.full((group * N_hexx, group * N_hexx), np.nan, dtype=complex)
    for g in range(group):
        for n in range(N):
            G_mat_interp_temp = G_matrix_interp[:, g * N + n]  # Extract a row
            print(f'Interpolating G_mat_interp_temp group {g+1}, position {n+1}')
            G_mat_interp_cols = interpolate_dPHI_rbf_1D(G_mat_interp_temp, group, N, map_detector, rbf_function='thin_plate_spline') # Perform interpolation on the column
            G_matrix_interp_cols[:, g * N + n] = G_mat_interp_cols  # Assign back to the row

    # --------------- UNFOLD GREEEN'S FUNCTION USING DIRECT METHOD -------------------
    print(f'Solve for dS using Direct Method')
    G_mat_interp_inverse = scipy.linalg.pinv(G_matrix_interp_cols)

    # UNFOLD ALL INTERPOLATED
    dS_unfold_INVERT = np.dot(G_mat_interp_inverse, dPHI_interp)

    # POSTPROCESS
    print(f'Postprocessing to appropriate dPHI')
    dS_unfold_INVERT_reshaped = np.reshape(dS_unfold_INVERT,(group,N))

    # Plotting of the neutron noise induced by the noise source #1 in the frequency domain
    for g in range(group):
        plt.figure()
        plt.plot(x, np.abs(S_reshaped[g]), 'g-', label=f'Group {g+1} - S_all')
        plt.plot(x, np.abs(dS_unfold_INVERT_reshaped[g]), 'r--', label=f'Group {g+1} - dS_allinterp')
        plt.legend()
        plt.ylabel('Normalized amplitude of the induced neutron noise')
        plt.title(f'Magnitude of neutron noise - Group {g+1}')
        plt.xlabel('Distance from core centre [cm]')
        plt.grid()
        plt.savefig(f'{output_dir}/{case_name}_03_INVERT/{case_name}_dS_INVERT_magnitude_G{g+1}.png')
        plt.close()

    # OUTPUT
    print(f'Generating JSON output for dS')
    output_direct1 = {}
    for g in range(group):
        dS_unfold_direct_groupname = f'dS_unfold{g+1}'
        dS_unfold_direct_list = [{"real": x.real, "imaginary": x.imag} for x in dS_unfold_INVERT_reshaped[g]]
        output_direct1[dS_unfold_direct_groupname] = dS_unfold_direct_list
    # Save data to JSON file
    with open(f'{output_dir}/{case_name}_03_INVERT/{case_name}_dS_INVERT_output.json', 'w') as json_file:
        json.dump(output_direct1, json_file, indent=4)
    
    return dPHI_INVERT, dS_unfold_INVERT

def main_unfold_1D_scan(dPHI_meas, dPHI, S, G_matrix, group, N, map_detector, output_dir, case_name, x):
    dPHI_reshaped = np.reshape(dPHI, (group, N))
    S_reshaped = np.reshape(S, (group, N))

##### 05. SCAN
    os.makedirs(f'{output_dir}/{case_name}_05_SCAN', exist_ok=True)

    # Create tuple of detector pairs
    flux_pos = [index for index, value in enumerate(map_detector) if value == 1]
    detector_pair = list(combinations(flux_pos, 2))
    delta_all = []
    for g in range(group):
        for n in range(N):
            m = g * N + n
            delta_AB = 0
            for p in range(len(detector_pair)):
                det_A = detector_pair[p][0]
                det_B = detector_pair[p][1]
                # Retrieve values for detectors A and B
                dPHI_A = dPHI_meas[g * N + (det_A)]
                dPHI_B = dPHI_meas[g * N + (det_B)]
                G_A = G_matrix[g * N + (det_A)][m]
                G_B = G_matrix[g * N + (det_B)][m]
                delta_AB += np.abs((dPHI_A / dPHI_B) - (G_A / G_B))
            delta_all.append(delta_AB)
            print(f'Done for group {g+1}, position {n}')

    # Save delta_all to a text file
    with open(f'{output_dir}/{case_name}_05_SCAN/{case_name}_delta_all.txt', 'w') as f:
        for item in delta_all:
            f.write(f"{item}\n")

    # Flatten dPHI_sol_temp_groups to a 1D list
    delta_all_full = delta_all
    delta_all_full_plot = np.reshape(delta_all_full, (group, N))  # 3D array, size (group, J_max, I_max)

    # Flatten dPHI_sol_temp_groups to a 1D list
    delta_all_full_inv = [1/x if x != 0 else np.inf for x in delta_all]
    delta_all_full_inv_plot = np.reshape(delta_all_full_inv, (group, N))  # 3D array, size (group, J_max, I_max)

    # Find minimum value and index
    min_value = min(delta_all)
    min_index = delta_all.index(min_value)
    for g in range(group):
        for n in range(N):
            if  g * N + n == min_index:
                print(f"Minimum value is {min_value} at index {min_index} (N = {n}) within group {g+1}")

    # Determine the scaling
    detector_loc = []
    for n in range(N):
        if map_detector[n] == 1:
            detector_loc.append(n)

    # Determine the scaling M
    G_sol_mat_new = G_matrix[detector_loc[0]][min_index]
    dPHI_meas_new = dPHI_meas[detector_loc[0]]
    M = dPHI_meas_new/G_sol_mat_new #np.abs(dPHI_meas_new/G_sol_mat_new)
    print(f'magnitude of dS unfold is {M}')

    dS_unfold_SCAN = [0.0] * group * N
    dS_unfold_SCAN[min_index] = M

    # Flatten dPHI_sol_temp_groups to a 1D list
    dS_unfold_SCAN_reshaped = np.reshape(dS_unfold_SCAN,(group,N))
    for g in range(group):
        plt.clf()
        plt.plot(x, np.abs(S_reshaped[g]), 'b-', label=f'Group {g+1} - dS Input')
        plt.plot(x, np.abs(dS_unfold_SCAN_reshaped[g]), 'r--', label=f'Group {g+1} - dS Unfold')
        plt.xlabel('height (cm)')
        plt.ylabel('magnitude of dS')
        plt.title('magnitude of noise unfolded')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{output_dir}/{case_name}_05_SCAN/{case_name}_dS_G{g+1}.png')
        plt.close()

    # OUTPUT
    print(f'Generating JSON output for dS')
    output = {}
    for g in range(group):
        dS_unfold_groupname = f'dS_unfold{g+1}'
        dS_unfold_list = [{"real": x.real, "imaginary": x.imag} for x in dS_unfold_SCAN_reshaped[g]]
        output[dS_unfold_groupname] = dS_unfold_list

    # Save data to JSON file
    with open(f'{output_dir}/{case_name}_05_SCAN/{case_name}_dS_output.json', 'w') as json_file:
        json.dump(output, json_file, indent=4)

    return dS_unfold_SCAN

#######################################################################################################
def main_unfold_1D_brute(dPHI_meas, dPHI, S, G_matrix, group, N, output_dir, case_name, x):
    dPHI_reshaped = np.reshape(dPHI, (group, N))
    S_reshaped = np.reshape(S, (group, N))

##### 06. BRUTE FORCE
    # --------------- MANIPULATE dPHI -------------------
    os.makedirs(f'{output_dir}/{case_name}_06_BRUTE', exist_ok=True)
    non_zero_indices = np.nonzero(dPHI_meas)[0]
    dPHI_meas = np.array(dPHI_meas)
    dPHI_meas_reshaped = np.reshape(dPHI_meas, (group, N))

    # Create dictionary to store each atom (column of G_matrix_full)
    G_dictionary = {}
    for g in range(group):
        for n in range(N):
            m = g * N + n
            G_dictionary[f"G_g{g+1}_n{n+1}"] = G_matrix[:, m]

    # Sample the dictionary atoms at known points (sparse form)
    G_dictionary_sampled = {k: np.zeros_like(dPHI_meas, dtype=complex) for k in G_dictionary}
    for k, o in G_dictionary.items():
        G_dictionary_sampled[k][non_zero_indices] = o[non_zero_indices]

    # Initialize variables
    valid_solution_BRUTE = False  # Flag to indicate a valid solution
    tol = 1E-8

    # Brute force over all combinations of atoms
    atom_keys = list(G_dictionary_sampled.keys())
    num_atoms = len(atom_keys)
    residual = 1
    residual_norm = 1
    chunk_size = 10000

    # Iterate over subsets of atoms
    for num_source in range(1, num_atoms + 1):
        print(f"Trying number of source mesh = {num_source}")
        iter_BRUTE = 0
        total_combinations = comb(num_atoms, num_source)

        subset_iter = combinations(atom_keys, num_source)

        while True:
            chunk = list(islice(subset_iter, chunk_size))
            if not chunk:
                break

            for subset in chunk:
                if iter_BRUTE % (20 * num_atoms) == 0:
                    percent = (iter_BRUTE / total_combinations) * 100 if total_combinations > 0 else 0
                    print(f"Iteration = {iter_BRUTE}, subset progress = {percent:.2f}%, subset = {subset}")

                # Form the initial matrix with the subset
                A = np.column_stack([G_dictionary_sampled[k] for k in subset]) #A = np.array([G_dictionary_sampled[k] for k in subset]).T
                try:
                    coeffs, _, _, _ = lstsq(A, dPHI_meas)
                except Exception as e:
                    print(f"lstsq failed for subset {subset}: {e}")
                    continue
                residual = dPHI_meas - A @ coeffs
                residual_norm = np.linalg.norm(residual)

                # Check if residual norm meets tolerance
                if residual_norm < tol:
                    print(f'Subsets {subset} pass the residual tolerance')
                    valid_solution_BRUTE = True  # Criterion satisfied
                    print(f"Valid solution found with number of sources = {num_source} and atoms = {subset}.")
                    coefficients = dict(zip(subset, coeffs))
                    dPHI_BRUTE = sum(c * G_dictionary[k] for k, c in zip(subset, coeffs))
                    break  # Exit the outer loop

                if valid_solution_BRUTE:
                    break  # Exit the subset loop

                iter_BRUTE += 1

            if valid_solution_BRUTE:
                break  # Exit the outer loop

        if valid_solution_BRUTE:
            break  # Exit the outer loop

    ###################################################################################################
    if valid_solution_BRUTE:
        # Reshape reconstructed signal
        dPHI_BRUTE_reshaped = np.reshape(dPHI_BRUTE, (group, N))

        # Plot results
        for g in range(group):
            plt.figure()
            plt.plot(x, np.abs(dPHI_reshaped[g])/np.max(np.abs(dPHI_reshaped[0])), 'g-', label=f'Group {g+1} - dPHI_sol')
            plt.plot(x, np.abs(dPHI_BRUTE_reshaped[g])/np.max(np.abs(dPHI_BRUTE_reshaped[0])), 'r--', label=f'Group {g+1} - dPHI_BRUTE')
            plt.scatter(x, np.abs(dPHI_meas_reshaped[g])/np.max(np.abs(dPHI_reshaped[0])), color='blue', label=f'Group {g+1} - dPHI_meas')
            plt.legend()
            plt.ylabel('Normalized amplitude of the induced neutron noise')
            plt.title(f'Magnitude of neutron noise - Group {g+1}')
            plt.xlabel('Distance from core centre [cm]')
            plt.grid()
            plt.savefig(f'{output_dir}/{case_name}_06_BRUTE/{case_name}_dPHI_BRUTE_magnitude_G{g+1}.png')
            plt.close()

        for g in range(group):
            plt.figure()
            plt.plot(x, np.degrees(np.angle(dPHI_reshaped[g])), 'g-', label=f'Group {g+1} - dPHI_sol')
            plt.plot(x, np.degrees(np.angle(dPHI_BRUTE_reshaped[g])), 'r--', label=f'Group {g+1} - dPHI_BRUTE')
            plt.legend()
            plt.ylabel('Phase of the induced neutron noise')
            plt.title(f'Phase of neutron noise - Group {g+1}')
            plt.xlabel('Distance from core centre [cm]')
            plt.grid()
            plt.savefig(f'{output_dir}/{case_name}_06_BRUTE/{case_name}_dPHI_BRUTE_phase_G{g+1}.png')
            plt.close()

        for g in range(group):
            plt.figure()
            plt.plot(x, dPHI_reshaped[g].real, 'g-', label=f'Group {g+1} - dPHI_sol')
            plt.plot(x, dPHI_BRUTE_reshaped[g].real, 'r--', label=f'Group {g+1} - dPHI_BRUTE')
            plt.scatter(x, dPHI_meas_reshaped[g].real, color='blue', label=f'Group {g+1} - dPHI_meas')
            plt.legend()
            plt.ylabel('Real component of the induced neutron noise')
            plt.title(f'Real component of neutron noise - Group {g+1}')
            plt.xlabel('Distance from core centre [cm]')
            plt.grid()
            plt.savefig(f'{output_dir}/{case_name}_06_BRUTE/{case_name}_dPHI_BRUTE_real_G{g+1}.png')
            plt.close()

        print("Matching pursuit completed and plots saved.")

        ######################################################################################################
        # --------------- UNFOLD GREEEN'S FUNCTION USING DIRECT METHOD -------------------
        print(f'Solve for dS using Direct Method')
        G_inverse = scipy.linalg.inv(G_matrix)

        # UNFOLD ALL INTERPOLATED
        dS_unfold_BRUTE = np.dot(G_inverse, dPHI_BRUTE)

        # POSTPROCESS
        print(f'Postprocessing to appropriate dPHI')
        dS_unfold_BRUTE_reshaped = np.reshape(dS_unfold_BRUTE,(group,N))

        # Plotting of the neutron noise induced by the noise source #1 in the frequency domain
        for g in range(group):
            plt.figure()
            plt.plot(x, np.abs(S_reshaped[g]), 'g-', label=f'Group {g+1} - S_all')
            plt.plot(x, np.abs(dS_unfold_BRUTE_reshaped[g]), 'r--', label=f'Group {g+1} - dS_unfold_BRUTE')
            plt.legend()
            plt.ylabel('Normalized amplitude of the induced neutron noise')
            plt.title(f'Magnitude of neutron noise - Group {g+1}')
            plt.xlabel('Distance from core centre [cm]')
            plt.grid()
            plt.savefig(f'{output_dir}/{case_name}_06_BRUTE/{case_name}_dS_BRUTE_magnitude_G{g+1}.png')
            plt.close()

        # OUTPUT
        print(f'Generating JSON output for dS')
        output_direct1 = {}
        for g in range(group):
            dS_unfold_direct_groupname = f'dS_unfold{g+1}'
            dS_unfold_direct_list = [{"real": x.real, "imaginary": x.imag} for x in dS_unfold_BRUTE_reshaped[g]]
            output_direct1[dS_unfold_direct_groupname] = dS_unfold_direct_list

        # Save data to JSON file
        with open(f'{output_dir}/{case_name}_06_BRUTE/{case_name}_dS_unfold_BRUTE_output.json', 'w') as json_file:
            json.dump(output_direct1, json_file, indent=4)

    else:
        print("No valid solution found with brute force.")

    return dPHI_BRUTE, dS_unfold_BRUTE

#######################################################################################################
def main_unfold_1D_back(dPHI_meas, dPHI, S, G_matrix, group, N, output_dir, case_name, x):
    dPHI_reshaped = np.reshape(dPHI, (group, N))
    S_reshaped = np.reshape(S, (group, N))

#### 07. BACKWARD ELIMINATION
    # --------------- MANIPULATE dPHI -------------------
    os.makedirs(f'{output_dir}/{case_name}_07_BACK', exist_ok=True)
    non_zero_indices = np.nonzero(dPHI_meas)[0]
    dPHI_meas = np.array(dPHI_meas)
    dPHI_meas_reshaped = np.reshape(dPHI_meas, (group, N))

    # Create dictionary to store each atom (column of G_matrix_full)
    G_dictionary = {}
    for g in range(group):
        for n in range(N):
            m = g * N + n
            G_dictionary[f"G_g{g+1}_n{n+1}"] = G_matrix[:, m]

    # Sample the dictionary atoms at known points (sparse form)
    G_dictionary_sampled = {k: np.zeros_like(dPHI_meas, dtype=complex) for k in G_dictionary}
    for k, o in G_dictionary.items():
        G_dictionary_sampled[k][non_zero_indices] = o[non_zero_indices]

    # Initialize variables for the higher loop
    valid_solution_BACK = False  # Flag to indicate a valid solution
    tol = 1E-10
    selected_atoms = list(G_dictionary_sampled.keys())
    residual = dPHI_meas.copy()
    iter_BACK = 0
    residual_norm = 1.0
    contribution_threshold = 1e-6  # Define the contribution threshold
    coefficients = {}

    # Dictionary to store valid solutions with term counts
    valid_solutions_BACK = {}

    #while not valid_solution:
    while selected_atoms:
        iter_BACK += 1
        print(f"Iteration {iter_BACK}: Atoms remaining = {len(selected_atoms)}")

        try:
            # Stack all selected atoms into the matrix
            A = np.array([G_dictionary_sampled[k] for k in selected_atoms]).T
            coeffs = np.linalg.lstsq(A, dPHI_meas, rcond=None)[0]
            coefficients = dict(zip(selected_atoms, coeffs))
            residual = dPHI_meas - A @ coeffs
            residual_norm = np.linalg.norm(residual)

            # Compute contributions (absolute value of coefficients)
            contributions = {atom: abs(coeff) / max(abs(coeffs)) for atom, coeff in zip(selected_atoms, coeffs)}

            # Validate the reconstructed signal against the criterion
            if residual_norm < tol:
                print(f"Valid solution found, selected atoms = {selected_atoms}, residual norm = {residual_norm:.6e}")
                valid_solutions_BACK[iter] = selected_atoms[:]
                atom_to_remove = min(contributions, key=contributions.get)
                selected_atoms.remove(atom_to_remove)
            else:
                # Find the least contributing atom
                atom_to_remove = min(contributions, key=contributions.get)
                selected_atoms.remove(atom_to_remove)
                print(f"Criteria not met. Removing least contributing atom: {atom_to_remove}, Contribution = {contributions[atom_to_remove]:.6e}, residual norm = {residual_norm:.6e}")
                if len(selected_atoms) == 0:
                    break

        except np.linalg.LinAlgError:
            print("SVD did not converge, skipping this iteration.")
            sorted_contributions = sorted(contributions.items(), key=lambda x: x[1])
            second_least_atom = sorted_contributions[1][0]  # Get the atom with the second smallest contribution
            selected_atoms.remove(second_least_atom)
            continue  # Skip to the next iteration

    if valid_solutions_BACK:
        best_atom = min(valid_solutions_BACK, key=lambda k: len(valid_solutions_BACK[k]))
        print(f"The best valid solution is with atom {best_atom} with iteration number = {valid_solutions_BACK[best_atom]}.")
        valid_solution_BACK = valid_solutions_BACK[best_atom]

        A = np.array([G_dictionary_sampled[k] for k in valid_solution_BACK]).T
        coeffs = np.linalg.lstsq(A, dPHI_meas, rcond=None)[0]
        coefficients = dict(zip(valid_solution_BACK, coeffs))
        dPHI_BACK = sum(c * G_dictionary[k] for k, c in coefficients.items())
    else:
        print("Failed to find a valid solution within the maximum number of outer iterations.")

    ###################################################################################################
    if valid_solution_BACK:
        # Reshape reconstructed signal
        dPHI_BACK_reshaped = np.reshape(dPHI_BACK, (group, N))

        # Plot results
        for g in range(group):
            plt.figure()
            plt.plot(x, np.abs(dPHI_reshaped[g])/np.max(np.abs(dPHI_reshaped[0])), 'g-', label=f'Group {g+1} - dPHI_sol')
            plt.plot(x, np.abs(dPHI_BACK_reshaped[g])/np.max(np.abs(dPHI_BACK_reshaped[0])), 'r--', label=f'Group {g+1} - dPHI_BACK')
            plt.scatter(x, np.abs(dPHI_meas_reshaped[g])/np.max(np.abs(dPHI_reshaped[0])), color='blue', label=f'Group {g+1} - dPHI_meas')
            plt.legend()
            plt.ylabel('Normalized amplitude of the induced neutron noise')
            plt.title(f'Magnitude of neutron noise - Group {g+1}')
            plt.xlabel('Distance from core centre [cm]')
            plt.grid()
            plt.savefig(f'{output_dir}/{case_name}_07_BACK/{case_name}_dPHI_BACK_magnitude_G{g+1}.png')
            plt.close()

        for g in range(group):
            plt.figure()
            plt.plot(x, np.degrees(np.angle(dPHI_reshaped[g])), 'g-', label=f'Group {g+1} - dPHI_sol')
            plt.plot(x, np.degrees(np.angle(dPHI_BACK_reshaped[g])), 'r--', label=f'Group {g+1} - dPHI_BACK')
            plt.legend()
            plt.ylabel('Phase of the induced neutron noise')
            plt.title(f'Phase of neutron noise - Group {g+1}')
            plt.xlabel('Distance from core centre [cm]')
            plt.grid()
            plt.savefig(f'{output_dir}/{case_name}_07_BACK/{case_name}_dPHI_BACK_phase_G{g+1}.png')
            plt.close()

        for g in range(group):
            plt.figure()
            plt.plot(x, dPHI_reshaped[g].real, 'g-', label=f'Group {g+1} - dPHI_sol')
            plt.plot(x, dPHI_BACK_reshaped[g].real, 'r--', label=f'Group {g+1} - dPHI_BACK')
            plt.scatter(x, dPHI_meas_reshaped[g].real, color='blue', label=f'Group {g+1} - dPHI_meas')
            plt.legend()
            plt.ylabel('Real component of the induced neutron noise')
            plt.title(f'Real component of neutron noise - Group {g+1}')
            plt.xlabel('Distance from core centre [cm]')
            plt.grid()
            plt.savefig(f'{output_dir}/{case_name}_07_BACK/{case_name}_dPHI_BACK_real_G{g+1}.png')
            plt.close()

        print("Matching pursuit completed and plots saved.")

        ######################################################################################################
        # --------------- UNFOLD GREEEN'S FUNCTION USING DIRECT METHOD -------------------
        print(f'Solve for dS using Direct Method')
        G_inverse = scipy.linalg.inv(G_matrix)

        # UNFOLD ALL INTERPOLATED
        dS_unfold_BACK = np.dot(G_inverse, dPHI_BACK)

        # POSTPROCESS
        print(f'Postprocessing to appropriate dPHI')
        dS_unfold_BACK_reshaped = np.reshape(dS_unfold_BACK,(group,N))

        # Plotting of the neutron noise induced by the noise source #1 in the frequency domain
        for g in range(group):
            plt.figure()
            plt.plot(x, np.abs(S_reshaped[g]), 'g-', label=f'Group {g+1} - S_all')
            plt.plot(x, np.abs(dS_unfold_BACK_reshaped[g]), 'r--', label=f'Group {g+1} - dS_unfold_BACK')
            plt.legend()
            plt.ylabel('Normalized amplitude of the induced neutron noise')
            plt.title(f'Magnitude of neutron noise - Group {g+1}')
            plt.xlabel('Distance from core centre [cm]')
            plt.grid()
            plt.savefig(f'{output_dir}/{case_name}_07_BACK/{case_name}_dS_BACK_magnitude_G{g+1}.png')
            plt.close()

        # OUTPUT
        print(f'Generating JSON output for dS')
        output_direct1 = {}
        for g in range(group):
            dS_unfold_direct_groupname = f'dS_unfold{g+1}'
            dS_unfold_direct_list = [{"real": x.real, "imaginary": x.imag} for x in dS_unfold_BACK_reshaped[g]]
            output_direct1[dS_unfold_direct_groupname] = dS_unfold_direct_list

        # Save data to JSON file
        with open(f'{output_dir}/{case_name}_07_BACK/{case_name}_dS_unfold_BACK_output.json', 'w') as json_file:
            json.dump(output_direct1, json_file, indent=4)

    else:
        print("No valid solution found with brute force.")

    return dPHI_BACK, dS_unfold_BACK

#######################################################################################################
def main_unfold_1D_greedy_optimized(dPHI_meas, dPHI, S, G_matrix, group, N, output_dir, case_name, x):
    dPHI_reshaped = np.reshape(dPHI, (group, N))
    S_reshaped = np.reshape(S, (group, N))
    non_zero_indices = np.nonzero(dPHI_meas)[0]
    dPHI_meas = np.array(dPHI_meas)
    dPHI_meas_reshaped = np.reshape(dPHI_meas, (group, N))

    # Create dictionary to store each atom (column of G_matrix_full)
    G_dictionary = {}
    for g in range(group):
        for n in range(N):
            m = g * N + n
            G_dictionary[f"G_g{g+1}_n{n+1}"] = G_matrix[:, m]

    # Sample the dictionary atoms at known points (sparse form)
    G_dictionary_sampled = {k: np.zeros_like(dPHI_meas, dtype=complex) for k in G_dictionary}
    for k, o in G_dictionary.items():
        G_dictionary_sampled[k][non_zero_indices] = o[non_zero_indices]

    keys = list(G_dictionary_sampled.keys())
    key_to_idx = {k: i for i, k in enumerate(keys)}
    G_full = np.column_stack([G_dictionary[k] for k in keys]).astype(complex, copy=False)#  (full, non‑sampled dictionary)
    X_full = np.column_stack([G_dictionary_sampled[k] for k in keys]).astype(complex, copy=False)
    y_full = np.asarray(dPHI_meas, dtype=complex)   
    obs = np.asarray(non_zero_indices)
    X = X_full[obs, :]          # shape (M_obs, N)
    y_dPHI = y_full[obs]             # shape (M_obs,)
    num_atoms = X.shape[1]      # N

##### 08. GREEDY
    os.makedirs(f'{output_dir}/{case_name}_08_GREEDY_OPTIMIZED', exist_ok=True)

    # Initialize variables for the higher loop
    valid_solution_GREEDY = False  # Flag to indicate a valid solution
    outer_iter = 0
    tol_GREEDY = 1E-10  # Stopping tolerance
    comb_first_atom = 1
    contribution_threshold = 1e-6  # Define the contribution threshold
    selected_atoms = []

    # Build all possible "first atom" combinations once
    first_atom_indices_iter = list(combinations(range(num_atoms), comb_first_atom))

    # Dictionary to store valid solutions with term counts
    valid_solutions_GREEDY = {}

    # Precompute column norms once outside all loops:
    col_norm_sq = np.einsum('ij,ij->j', X.conj(), X).real  # shape (N,)

    # ---------- Outer loop over first atom(s) ----------
    for outer_iter, first_atom_idx_tuple in enumerate(first_atom_indices_iter, start=1):

        # Initialize residual and selected atoms
        selected_idx = list(first_atom_idx_tuple)  # indices into columns of X
        A = X[:, selected_idx]                     # view, no copy
        
        # Solve LS and residual
        coeffs, *_ = np.linalg.lstsq(A, y_dPHI, rcond=None)
        residual = y_dPHI - A @ coeffs
        residual_norm = np.linalg.norm(residual)

        first_atom_keys = tuple(keys[j] for j in first_atom_idx_tuple)
        print(f"Outer iteration {outer_iter}: Trying first atom {first_atom_keys}, "
          f"current residual norm = {residual_norm:.6e}")

        prev_selected_len = 0
        constant_len_counter = 0

        # ---------- Inner greedy loop (QR-based exact scoring, no per-candidate LS) ----------
        while residual_norm > tol_GREEDY:
            # Build boolean mask / remaining indices
            selected_mask = np.zeros(num_atoms, dtype=bool)
            selected_mask[selected_idx] = True
            remaining = np.flatnonzero(~selected_mask)
            if remaining.size < comb_first_atom:
                break
            if comb_first_atom != 1:
                raise NotImplementedError("QR shortcut shown here assumes comb_first_atom == 1")

            # Economy QR of current A (once per iteration)
            if len(selected_idx) == 0:
                # No columns selected yet: Q is empty, so QQ^H = 0
                Q = None
                # Current residual r is y_dPHI itself
                r = residual  # already y_dPHI in your first step
                rn_sq = residual_norm**2
                # For scoring: denom_j = ‖x_j‖², u = X_remᴴ @ r
                X_rem = X[:, remaining]  # view
                u = X_rem.conj().T @ r                         # (R,)
                denom = col_norm_sq[remaining].copy()          # (R,)
            else:
                # Compute Q of A = X[:, selected_idx]
                Q, _ = np.linalg.qr(X[:, selected_idx], mode='reduced')  # Q: (M_obs, k)
                r = residual - Q @ (Q.conj().T @ residual)
                rn_sq = float(np.vdot(r, r).real)

                # Batch projections
                X_rem = X[:, remaining]                        # (M_obs, R)
                u = X_rem.conj().T @ r                         # (R,)   correlations with residual
                B = Q.conj().T @ X_rem                         # (k, R) projection onto current subspace
                denom = col_norm_sq[remaining] - np.sum(np.abs(B)**2, axis=0)  # (R,)

            # Avoid division by zero / nearly dependent candidates
            eps = 1e-15
            valid = denom > eps
            if not np.any(valid):
                print("   No linearly independent candidates left. Stopping.")
                break
            
            # Exact residual for each candidate after refit
            rn_sq_candidates = np.full(remaining.shape, np.inf, dtype=float)
            rn_sq_candidates[valid] = rn_sq - (np.abs(u[valid])**2) / denom[valid]

            # Pick best candidate (min residual)
            best_pos = int(np.argmin(rn_sq_candidates))
            best_j = int(remaining[best_pos])
            best_rn = float(np.sqrt(max(rn_sq_candidates[best_pos], 0.0)))

            # Add chosen atom
            selected_idx.append(best_j)

            # Refit with updated set
            A = X[:, selected_idx]
            coeffs, *_ = np.linalg.lstsq(A, y_dPHI, rcond=None)
            residual = y_dPHI - A @ coeffs
            residual_norm = np.linalg.norm(residual)

        # Check for low contribution atoms
        keep_pruning = True
        while keep_pruning and len(selected_idx) > 1:
        
            residual_effects = {}   # atom_key -> residual norm after removal
    
            # Try removing each atom and compute residual effect
            for i, atom in enumerate(selected_idx):
                trial_idx = selected_idx[:i] + selected_idx[i+1:]
                A_trial = X[:, trial_idx]
                coeffs_trial, *_ = np.linalg.lstsq(A_trial, y_dPHI, rcond=None)
                r_trial = y_dPHI - A_trial @ coeffs_trial
                rn_trial = np.linalg.norm(r_trial)
    
                # Save effect
                residual_effects[atom] = rn_trial
    
            # Find atom whose removal results in the MINIMAL degradation
            atom_to_remove = min(residual_effects.keys(), key=lambda a: residual_effects[a])
            best_rn_after_removal = residual_effects[atom_to_remove]
    
            # Condition to accept removal
            if best_rn_after_removal < tol_GREEDY:
                print(f"   Pruning atom {keys[atom_to_remove]}: new residual = {best_rn_after_removal:.3e}")
    
                # Perform the removal
                selected_idx.remove(atom_to_remove)
                A = X[:, selected_idx]
                coeffs, *_ = np.linalg.lstsq(A, y_dPHI, rcond=None)
                residual = y_dPHI - A @ coeffs
                residual_norm = np.linalg.norm(residual)
    
            else:
                # Cannot remove any atom safely → stop pruning
                keep_pruning = False
                print(f"   Cannot prune any more atoms without exceeding tolerance. Stopping pruning.")

        # ---------- Validate and store ----------
        if residual_norm < tol_GREEDY:
            valid_solution_GREEDY = True
            print(f"Valid solution found with first atom {first_atom_keys} in outer iteration {outer_iter}.")
            valid_solutions_GREEDY[first_atom_keys] = [keys[j] for j in selected_idx]
        else:
            print(f"Criterion not met with first atom {first_atom_keys}. Restarting with a new atom.")

    # Final check for the best solution
    if valid_solutions_GREEDY:
        best_atom = min(valid_solutions_GREEDY, key=lambda k: len(valid_solutions_GREEDY[k]))
        print(f"\nThe best valid solution is with atom {best_atom} with selected atoms = {valid_solutions_GREEDY[best_atom]}.")
        valid_solution_GREEDY = valid_solutions_GREEDY[best_atom]

        A = np.array([G_dictionary_sampled[k] for k in valid_solution_GREEDY]).T
        coeffs = np.linalg.lstsq(A, dPHI_meas, rcond=None)[0]
        coefficients = dict(zip(valid_solution_GREEDY, coeffs))
        dPHI_GREEDY = sum(c * G_dictionary[k] for k, c in zip(valid_solution_GREEDY, coeffs))

    ####################################################################################################
    if valid_solution_GREEDY:
        # Reshape reconstructed signal
        dPHI_GREEDY_reshaped = np.reshape(dPHI_GREEDY, (group, N))

        # Plot results
        for g in range(group):
            plt.figure()
            plt.plot(x, np.abs(dPHI_reshaped[g])/np.max(np.abs(dPHI_reshaped[0])), 'g-', label=f'Group {g+1} - dPHI_sol')
            plt.plot(x, np.abs(dPHI_GREEDY_reshaped[g])/np.max(np.abs(dPHI_GREEDY_reshaped[0])), 'r--', label=f'Group {g+1} - dPHI_GREEDY')
            plt.scatter(x, np.abs(dPHI_meas_reshaped[g])/np.max(np.abs(dPHI_reshaped[0])), color='blue', label=f'Group {g+1} - dPHI_meas')
            plt.legend()
            plt.ylabel('Normalized amplitude of the induced neutron noise')
            plt.title(f'Magnitude of neutron noise - Group {g+1}')
            plt.xlabel('Distance from core centre [cm]')
            plt.grid()
            plt.savefig(f'{output_dir}/{case_name}_08_GREEDY_OPTIMIZED/{case_name}_dPHI_GREEDY_magnitude_G{g+1}.png')
            plt.close()

        for g in range(group):
            plt.figure()
            plt.plot(x, np.degrees(np.angle(dPHI_reshaped[g])), 'g-', label=f'Group {g+1} - dPHI_sol')
            plt.plot(x, np.degrees(np.angle(dPHI_GREEDY_reshaped[g])), 'r--', label=f'Group {g+1} - dPHI_GREEDY')
            plt.legend()
            plt.ylabel('Phase of the induced neutron noise')
            plt.title(f'Phase of neutron noise - Group {g+1}')
            plt.xlabel('Distance from core centre [cm]')
            plt.grid()
            plt.savefig(f'{output_dir}/{case_name}_08_GREEDY_OPTIMIZED/{case_name}_dPHI_GREEDY_phase_G{g+1}.png')
            plt.close()

        for g in range(group):
            plt.figure()
            plt.plot(x, dPHI_reshaped[g].real, 'g-', label=f'Group {g+1} - dPHI_sol')
            plt.plot(x, dPHI_GREEDY_reshaped[g].real, 'r--', label=f'Group {g+1} - dPHI_GREEDY')
            plt.scatter(x, dPHI_meas_reshaped[g].real, color='blue', label=f'Group {g+1} - dPHI_meas')
            plt.legend()
            plt.ylabel('Real component of the induced neutron noise')
            plt.title(f'Real component of neutron noise - Group {g+1}')
            plt.xlabel('Distance from core centre [cm]')
            plt.grid()
            plt.savefig(f'{output_dir}/{case_name}_08_GREEDY_OPTIMIZED/{case_name}_dPHI_GREEDY_real_G{g+1}.png')
            plt.close()

        ######################################################################################################
        # --------------- UNFOLD GREEEN'S FUNCTION USING DIRECT METHOD -------------------
        print(f'\nSolve for dS using Direct Method')
        G_inverse = scipy.linalg.inv(G_matrix)

        # UNFOLD ALL INTERPOLATED
        dS_unfold_GREEDY = np.dot(G_inverse, dPHI_GREEDY)

        # POSTPROCESS
        print(f'Postprocessing to appropriate dPHI')
        dS_unfold_GREEDY_reshaped = np.reshape(dS_unfold_GREEDY,(group,N))

        # Plotting of the neutron noise induced by the noise source #1 in the frequency domain
        for g in range(group):
            plt.figure()
            plt.plot(x, np.abs(S_reshaped[g]), 'g-', label=f'Group {g+1} - S_all')
            plt.plot(x, np.abs(dS_unfold_GREEDY_reshaped[g]), 'r--', label=f'Group {g+1} - dS_unfold_GREEDY')
            plt.legend()
            plt.ylabel('Normalized amplitude of the induced neutron noise')
            plt.title(f'Magnitude of neutron noise - Group {g+1}')
            plt.xlabel('Distance from core centre [cm]')
            plt.grid()
            plt.savefig(f'{output_dir}/{case_name}_08_GREEDY_OPTIMIZED/{case_name}_dS_GREEDY_magnitude_G{g+1}.png')
            plt.close()

        # OUTPUT
        print(f'Generating JSON output for dS')
        output_direct1 = {}
        for g in range(group):
            dS_unfold_direct_groupname = f'dS_unfold{g+1}'
            dS_unfold_direct_list = [{"real": x.real, "imaginary": x.imag} for x in dS_unfold_GREEDY_reshaped[g]]
            output_direct1[dS_unfold_direct_groupname] = dS_unfold_direct_list

        # Save data to JSON file
        with open(f'{output_dir}/{case_name}_08_GREEDY_OPTIMIZED/{case_name}_dS_unfold_GREEDY_output.json', 'w') as json_file:
            json.dump(output_direct1, json_file, indent=4)

    else:
        print("Failed to find a valid solution within the maximum number of outer iterations.")

    return dPHI_GREEDY, dS_unfold_GREEDY

