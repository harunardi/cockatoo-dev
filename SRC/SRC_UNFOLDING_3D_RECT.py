import numpy as np
import json
import time
import os
import sys
import scipy.linalg
from itertools import combinations, islice
from math import comb
from petsc4py import PETSc
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
from SRC.XSPROCESS_3D_RECT import *

#######################################################################################################
def main_unfold_3D_rect_noise(PHI_temp, keff, group, N, I_max, J_max, K_max, TOT, SIGS_reshaped, BC, dx, dy, dz, D, chi, NUFIS, precond, v, Beff, omega, l, dTOT, dSIGS_reshaped, dNUFIS, map_detector, output_dir, case_name, x, y, z):

##### Noise Simulation
    solver_type = 'noise'
    os.makedirs(f'{output_dir}/{case_name}_{solver_type.upper()}', exist_ok=True)
    conv = convert_index_3D_rect(D, I_max, J_max, K_max)
    conv_array = np.array(conv)
    max_conv = max(conv)

    PHI = PHI_temp.copy()
    
    matrix_builder = MatrixBuilderNoise3DRect(group, N, conv, TOT, SIGS_reshaped, BC, dx, dy, dz, D, chi, NUFIS, keff, v, Beff, omega, l, dTOT, dSIGS_reshaped, dNUFIS)
    M, dS = matrix_builder.build_noise_matrices()

    solver = SolverFactory.get_solver_fixed3DRect(solver_type, group, N, conv, M, dS, PHI, dx, dy, dz, precond, tol=1e-10)

    dPHI_temp = solver.solve()
    dPHI, dPHI_reshaped, dPHI_reshaped_plot = PostProcessor.postprocess_fixed3DRect(dPHI_temp, conv, group, N, I_max, J_max, K_max)

    output = {}
    for g in range(len(dPHI_reshaped)):
        dPHI_groupname = f'dPHI{g + 1}'
        dPHI_list = [{"real": x.real, "imaginary": x.imag} for x in dPHI_reshaped[g]]
        output[dPHI_groupname] = dPHI_list

    with open(f'{output_dir}/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_output.json', 'w') as json_file:
        json.dump(output, json_file, indent=4)

    return dPHI_temp

def main_unfold_3D_rect_green(PHI_temp, keff, group, N, I_max, J_max, K_max, TOT, SIGS_reshaped, BC, dx, dy, dz, D, chi, NUFIS, precond, v, Beff, omega, l, dTOT, dSIGS_reshaped, dNUFIS, map_detector, output_dir, case_name, x, y, z):

    conv = convert_index_3D_rect(D, I_max, J_max, K_max)
    conv_array = np.array(conv)
    max_conv = max(conv)
    PHI = PHI_temp.copy()

##### 01. Green's Function Generation
    os.makedirs(f'{output_dir}/{case_name}_01_GENERATE', exist_ok=True)
    matrix_builder = MatrixBuilderNoise3DRect(group, N, conv, TOT, SIGS_reshaped, BC, dx, dy, dz, D, chi, NUFIS, keff, v, Beff, omega, l, dTOT, dSIGS_reshaped, dNUFIS)
    M, dS = matrix_builder.build_noise_matrices()

    M_petsc = PETSc.Mat().createAIJ(size=M.shape, csr=(M.indptr, M.indices, M.data), comm=PETSc.COMM_WORLD)
    M_petsc.assemble()

    # PETSc Solver (KSP) and Preconditioner (PC)
    ksp = PETSc.KSP().create()
    ksp.setOperators(M_petsc)
    ksp.setType(PETSc.KSP.Type.GMRES)

    pc = ksp.getPC()
    if precond == 0:
        print(f'Solving using Sparse Solver')
        pc.setType(PETSc.PC.Type.NONE)
    elif precond == 1:
        print(f'Solving using ILU')
        pc.setType(PETSc.PC.Type.ILU)
        print(f'ILU Preconditioner Done')
    elif precond == 2:
        print('Solving using LU Decomposition')
        pc.setType(PETSc.PC.Type.LU)
        print(f'LU Preconditioner Done')

    # Solver tolerances
    ksp.setTolerances(rtol=1e-10, max_it=5000)

    G_sol_all = np.ones(group*N, dtype=complex)
    G_sol_temp = np.ones(group*max_conv, dtype=complex)
    G_matrix = np.zeros((group * max_conv, group * max_conv), dtype=complex)
    
    for g in range(group):
        for n in range(N):
            if conv[n] != 0:
                dS = [0] * (group * max_conv)
                dS[g*max_conv+(conv[n]-1)] = 1  # Set the relevant entry to 1
                dS_petsc = PETSc.Vec().createWithArray(dS)
                dS_petsc.assemble()

                errdPHI = 1
                tol = 1E-10
                iter = 0

                while errdPHI > tol:
                    G_sol_tempold = np.copy(G_sol_temp)
                    G_sol_temp_petsc = PETSc.Vec().createWithArray(G_sol_temp)

                    # Solve the linear system using PETSc KSP
                    ksp.solve(dS_petsc, G_sol_temp_petsc)

                    # Get result back into NumPy array
                    G_sol_temp = G_sol_temp_petsc.getArray()

                    errdPHI = np.max(np.abs(G_sol_temp - G_sol_tempold) / (np.abs(G_sol_temp) + 1E-20))

                for gp in range(group):
                    for m in range(N):
                        G_sol_all[gp * N + m] = G_sol_temp[gp * max_conv + (conv[m] - 1)]
                G_sol_reshape = np.reshape(G_sol_all, (group, N))
                G_matrix[:, g*max_conv+(conv[n]-1)] = G_sol_temp.flatten()  # Assign solution to row
                
                # OUTPUT
                output = {}
                for gp in range(group):
                    G_sol_groupname = f'G{g+1}{gp+1}'
                    G_sol_list = [{"real": x.real, "imaginary": x.imag} for x in G_sol_reshape[gp]]
                    output[G_sol_groupname] = G_sol_list

                i = n % I_max
                j = (n // I_max) % J_max
                k = n // (I_max * J_max)

               # Save output to HDF5 file
                hdf5_filename = f'{output_dir}/{case_name}_01_GENERATE/Green_g{g+1}_k{k+1}_j{j+1}_i{i+1}.h5'
                save_output_hdf5(hdf5_filename, output)
                print(f'Generated Green Function for group = {g + 1}, K = {k+1}, J = {j+1}, I = {i+1}')

    return G_matrix

def main_unfold_3D_rect_solve(PHI_temp, G_matrix, dPHI_temp, keff, group, N, I_max, J_max, K_max, TOT, SIGS_reshaped, BC, dx, dy, dz, D, chi, NUFIS, precond, v, Beff, omega, l, dTOT, dSIGS_reshaped, dNUFIS, map_detector, output_dir, case_name, x, y, z):

    conv = convert_index_3D_rect(D, I_max, J_max, K_max)
    conv_array = np.array(conv)
    max_conv = max(conv)
    PHI = PHI_temp.copy()

##### 02. Solve
    os.makedirs(f'{output_dir}/{case_name}_02_SOLVE', exist_ok=True)
    output_SOLVE = f'{output_dir}/{case_name}_02_SOLVE/{case_name}'

    with h5py.File(f'{output_dir}/{case_name}_02_SOLVE/{case_name}_G_matrix_full.h5', 'w') as hf:
        hf.create_dataset('G_matrix', data=G_matrix)
    plt.figure(figsize=(8, 6))
    plt.imshow(G_matrix.real, cmap='viridis', interpolation='nearest', origin='lower')
    plt.colorbar(label='Magnitude of G_matrix_full')
    plt.xlabel('Index')
    plt.ylabel('Index')
    plt.gca().invert_yaxis()
    plt.title('Plot of the Magnitude of G_matrix_full')
    plt.savefig(f'{output_dir}/{case_name}_02_SOLVE/{case_name}_G_matrix_full.png')

    matrix_builder = MatrixBuilderNoise3DRect(group, N, conv, TOT, SIGS_reshaped, BC, dx, dy, dz, D, chi, NUFIS, keff, v, Beff, omega, l, dTOT, dSIGS_reshaped, dNUFIS)
    M, dS = matrix_builder.build_noise_matrices()
    S = dS.dot(PHI)

    # Iterate over group pairs to compute dPHI
    dPHI_temp_SOLVE = np.zeros((group * max_conv), dtype=complex)
    for i in range(group):
        for j in range(group):
            # Extract the relevant blocks from G_matrix and S
            G_block = G_matrix[i*max_conv:(i+1)*max_conv, j*max_conv:(j+1)*max_conv]
            S_block = S[j*max_conv:(j+1)*max_conv]
        
            # Perform the matrix-vector multiplication for the Green's function
            dPHI_temp_SOLVE[i*max_conv:(i+1)*max_conv] += np.dot(G_block, S_block)

    non_zero_indices = np.nonzero(conv)[0]
    dPHI_temp_indices = conv_array[non_zero_indices] - 1
    dPHI_SOLVE = np.zeros((group * N), dtype=complex)
    S_all = np.zeros((group * N), dtype=complex)
    for g in range(group):
        dPHI_temp_start = g * max(conv)
        dPHI_SOLVE[g * N + non_zero_indices] = dPHI_temp_SOLVE[dPHI_temp_start + dPHI_temp_indices]
        S_all[g * N + non_zero_indices] = S[dPHI_temp_start + dPHI_temp_indices]
        for n in range(N):
            if conv[n] == 0:
                dPHI_SOLVE[g*N+n] = np.nan
                S_all[g*N+n] = np.nan
    dPHI_SOLVE_reshaped = np.reshape(dPHI_SOLVE, (group, N))
    S_all_reshaped = np.reshape(S_all, (group, N))

    # OUTPUT
    print(f'Generating JSON output')
    output = {}
    for g in range(group):
        dPHI_SOLVE_groupname = f'dPHI{g+1}'
        dPHI_SOLVE_list = [{"real": x.real, "imaginary": x.imag} for x in dPHI_SOLVE_reshaped[g]]
        output[dPHI_SOLVE_groupname] = dPHI_SOLVE_list

        S_groupname = f'S{g+1}'
        S_list = [{"real": x.real, "imaginary": x.imag} for x in S_all_reshaped[g]]
        output[S_groupname] = S_list

    # Save data to JSON file
    with open(f'{output_dir}/{case_name}_02_SOLVE/{case_name}_output.json', 'w') as json_file:
        json.dump(output, json_file, indent=4)

    dPHI_SOLVE_reshaped_plot = np.reshape(dPHI_SOLVE, (group, K_max, J_max, I_max))
    S_all_reshaped_plot = np.reshape(S_all, (group, K_max, J_max, I_max))
    for g in range(group):
        image_files_mag = []
        image_files_phase = []
        for k in range(K_max):
            filename_mag_dPHI_SOLVE = plot_heatmap_3D_general(dPHI_SOLVE_reshaped_plot[g, k, :, :], g+1, k+1, x, y, cmap='viridis', title=f'2D Plot of dPHI{g+1}_SOLVE, Z={k+1}', output=output_SOLVE, varname='dPHI_SOLVE', case_name=case_name, process_data='magnitude')
            filename_phase_dPHI_SOLVE = plot_heatmap_3D_general(dPHI_SOLVE_reshaped_plot[g, k, :, :], g+1, k+1, x, y, cmap='viridis', title=f'2D Plot of dPHI{g+1}_SOLVE, Z={k+1}', output=output_SOLVE, varname='dPHI_SOLVE', case_name=case_name, process_data='phase')
            image_files_mag.append(filename_mag_dPHI_SOLVE)
            image_files_phase.append(filename_phase_dPHI_SOLVE)
        gif_filename_mag_dPHI_SOLVE = f'{output_SOLVE}_magnitude_dPHI_SOLVE_animation_G{g+1}.gif'
        gif_filename_phase_dPHI_SOLVE = f'{output_SOLVE}_phase_dPHI_SOLVE_animation_G{g+1}.gif'
        images_mag_dPHI_SOLVE = [Image.open(img) for img in image_files_mag]
        images_phase_dPHI_SOLVE = [Image.open(img) for img in image_files_phase]
        images_mag_dPHI_SOLVE[0].save(gif_filename_mag_dPHI_SOLVE, save_all=True, append_images=images_mag_dPHI_SOLVE[1:], duration=300, loop=0)
        images_phase_dPHI_SOLVE[0].save(gif_filename_phase_dPHI_SOLVE, save_all=True, append_images=images_phase_dPHI_SOLVE[1:], duration=300, loop=0)

    for g in range(group):
        image_files_mag = []
        image_files_phase = []
        for k in range(K_max):
            filename_mag_S_all = plot_heatmap_3D_general(S_all_reshaped_plot[g, k, :, :], g+1, k+1, x, y, cmap='viridis', title=f'2D Plot of S{g+1}, Z={k+1}', output=output_SOLVE, varname='S', case_name=case_name, process_data='magnitude')
            filename_phase_S_all = plot_heatmap_3D_general(S_all_reshaped_plot[g, k, :, :], g+1, k+1, x, y, cmap='viridis', title=f'2D Plot of S{g+1}, Z={k+1}', output=output_SOLVE, varname='S', case_name=case_name, process_data='phase')
            image_files_mag.append(filename_mag_S_all)
            image_files_phase.append(filename_phase_S_all)
        gif_filename_mag_S_all = f'{output_SOLVE}_magnitude_S_all_animation_G{g+1}.gif'
        gif_filename_phase_S_all = f'{output_SOLVE}_phase_S_all_animation_G{g+1}.gif'
        images_mag_S_all = [Image.open(img) for img in image_files_mag]
        images_phase_S_all = [Image.open(img) for img in image_files_phase]
        images_mag_S_all[0].save(gif_filename_mag_S_all, save_all=True, append_images=images_mag_S_all[1:], duration=300, loop=0)
        images_phase_S_all[0].save(gif_filename_phase_S_all, save_all=True, append_images=images_phase_S_all[1:], duration=300, loop=0)

    # UNFOLDING
    G_inverse = scipy.linalg.inv(G_matrix)
    dS_unfold_temp_SOLVE = np.dot(G_inverse, dPHI_temp_SOLVE)
    dS_unfold_SOLVE = np.zeros((group * N), dtype=complex)  # Assuming N >= max_conv

    # POSTPROCESS
    print(f'Postprocessing to appropriate dPHI')
    non_zero_indices = np.nonzero(conv)[0]
    dS_unfold_temp_indices = conv_array[non_zero_indices] - 1

    for g in range(group):
        dS_unfold_temp_start = g * max(conv)
        dS_unfold_SOLVE[g * N + non_zero_indices] = dS_unfold_temp_SOLVE[dS_unfold_temp_start + dS_unfold_temp_indices]
        for n in range(N):
            if conv[n] == 0:
                dS_unfold_SOLVE[g*N+n] = np.nan
    dS_unfold_SOLVE_reshaped = np.reshape(dS_unfold_SOLVE,(group,N))

    # OUTPUT
    print(f'Generating JSON output for dS')
    output = {}
    for g in range(group):
        dS_unfold_SOLVE_groupname = f'dS_unfold{g+1}'
        dS_unfold_SOLVE_list = [{"real": x.real, "imaginary": x.imag} for x in dS_unfold_SOLVE_reshaped[g]]
        output[dS_unfold_SOLVE_groupname] = dS_unfold_SOLVE_list

    # Save data to JSON file
    with open(f'{output_dir}/{case_name}_02_SOLVE/{case_name}_dS_output.json', 'w') as json_file:
        json.dump(output, json_file, indent=4)

    dS_SOLVE_reshaped_plot = np.reshape(dS_unfold_SOLVE, (group, K_max, J_max, I_max))

    for g in range(group):
        image_files_mag = []
        image_files_phase = []
        for k in range(K_max):
            filename_mag_dS_SOLVE = plot_heatmap_3D_general(dS_SOLVE_reshaped_plot[g, k, :, :], g+1, k+1, x, y, cmap='viridis', title=f'2D Plot of dS{g+1}_SOLVE, Z={k+1}', output=output_SOLVE, varname='dS_SOLVE', case_name=case_name, process_data='magnitude')
            filename_phase_dS_SOLVE = plot_heatmap_3D_general(dS_SOLVE_reshaped_plot[g, k, :, :], g+1, k+1, x, y, cmap='viridis', title=f'2D Plot of dS{g+1}_SOLVE, Z={k+1}', output=output_SOLVE, varname='dS_SOLVE', case_name=case_name, process_data='phase')
            image_files_mag.append(filename_mag_dS_SOLVE)
            image_files_phase.append(filename_phase_dS_SOLVE)
        gif_filename_mag_dS_SOLVE = f'{output_SOLVE}_magnitude_dS_SOLVE_animation_G{g+1}.gif'
        gif_filename_phase_dS_SOLVE = f'{output_SOLVE}_phase_dS_SOLVE_animation_G{g+1}.gif'
        images_mag_dS_SOLVE = [Image.open(img) for img in image_files_mag]
        images_phase_dS_SOLVE = [Image.open(img) for img in image_files_phase]
        images_mag_dS_SOLVE[0].save(gif_filename_mag_dS_SOLVE, save_all=True, append_images=images_mag_dS_SOLVE[1:], duration=300, loop=0)
        images_phase_dS_SOLVE[0].save(gif_filename_phase_dS_SOLVE, save_all=True, append_images=images_phase_dS_SOLVE[1:], duration=300, loop=0)

    # Calculate error and compare
    diff_S1 = np.abs(np.array(dS_unfold_SOLVE_reshaped[0]) - np.array(S_all_reshaped[0]))
    diff_S2 = np.abs(np.array(dS_unfold_SOLVE_reshaped[1]) - np.array(S_all_reshaped[1]))
    diff_S = [[diff_S1], [diff_S2]]
    diff_S_array = np.array(diff_S)
    diff_S_reshaped = diff_S_array.reshape(group, K_max, J_max, I_max)

    for g in range(group):
        image_files_mag = []
        image_files_phase = []
        for k in range(K_max):
            filename_mag_diff_S = plot_heatmap_3D_general(diff_S_reshaped[g, k, :, :], g+1, k+1, x, y, cmap='viridis', title=f'2D Plot of diff_S{g+1}, Z={k+1}', output=output_SOLVE, varname='diff_S', case_name=case_name, process_data='magnitude')
            filename_phase_diff_S = plot_heatmap_3D_general(diff_S_reshaped[g, k, :, :], g+1, k+1, x, y, cmap='viridis', title=f'2D Plot of diff_S{g+1}, Z={k+1}', output=output_SOLVE, varname='diff_S', case_name=case_name, process_data='phase')
            image_files_mag.append(filename_mag_diff_S)
            image_files_phase.append(filename_phase_diff_S)
        gif_filename_mag_diff_S = f'{output_SOLVE}_magnitude_diff_S_animation_G{g+1}.gif'
        gif_filename_phase_diff_S = f'{output_SOLVE}_phase_diff_S_animation_G{g+1}.gif'
        images_mag_diff_S = [Image.open(img) for img in image_files_mag]
        images_phase_diff_S = [Image.open(img) for img in image_files_phase]
        images_mag_diff_S[0].save(gif_filename_mag_diff_S, save_all=True, append_images=images_mag_diff_S[1:], duration=300, loop=0)
        images_phase_diff_S[0].save(gif_filename_phase_diff_S, save_all=True, append_images=images_phase_diff_S[1:], duration=300, loop=0)

    # --------------- MANIPULATE dPHI -------------------
    # Define zeroed dPHI as dPHI_temp_zero and dPHI_zero
    dPHI_temp_meas = dPHI_temp.copy() # 1D list, size (group * max_conv)
    for g in range(group):
        for n in range(len(map_detector)):
            if map_detector[n] == 0:
                idx = g * max_conv + (conv[n]-1)
                dPHI_temp_meas[idx] = 0

    map_detector_conv = np.zeros((group * max_conv))
    for n in range(N):
        if conv[n] != 0:
            map_detector_conv[conv[n] - 1] = map_detector[n]

    map_det_S = np.zeros((group * max_conv))
    with open(f'{output_dir}/{case_name}_02_SOLVE/detector_source_report.txt', 'w') as f:
        for g in range(group):
            for n in range(max_conv):
                if S[g * max_conv + n] != 0:
                    line = f"For group {g+1}, mesh {n+1}, S = {S[g * max_conv + n]}\n"
                    f.write(line)
                    map_det_S[g * max_conv + n] += 1
                if map_detector_conv[n] == 1:
                    map_det_S[g * max_conv + n] += 0.5

    map_det_S_new = np.full((group * N), np.nan)  # start with nan

    for g in range(group):
        for n in range(N):
            if conv[n] != 0:
                map_det_S_new[g*N + n] = map_det_S[g*max_conv + conv[n] - 1]
    map_det_S_new_plot = np.reshape(map_det_S_new, (group, K_max, J_max, I_max))

    for g in range(group):
        image_files_mag = []
        for k in range(K_max):
            filename_mag_dS_SOLVE = plot_heatmap_3D_categorical(map_det_S_new_plot[g, k, :, :], g+1, k+1, x, y, output=output_SOLVE, varname='closeness_det_S', case_name=case_name, title=f'2D Plot of Closeness Detector-Source Group {g+1}, Z={k+1}')
            image_files_mag.append(filename_mag_dS_SOLVE)
        gif_filename_mag_dS_SOLVE = f'{output_SOLVE}_closeness_det_S_animation_G{g+1}.gif'
        images_mag_dS_SOLVE = [Image.open(img) for img in image_files_mag]
        images_mag_dS_SOLVE[0].save(gif_filename_mag_dS_SOLVE, save_all=True, append_images=images_mag_dS_SOLVE[1:], duration=300, loop=0)

    return S, dPHI_temp_meas

#######################################################################################################
def main_unfold_3D_rect_invert(dPHI_temp_meas, dPHI_temp, S, G_matrix, group, N, I_max, J_max, K_max, conv, map_detector, map_zone, output_dir, case_name, x, y, z):
    max_conv = max(conv)
    conv_array = np.array(conv)
    non_zero_conv = np.nonzero(conv)[0]
    dPHI_temp_conv = conv_array[non_zero_conv] - 1
    dPHI = np.zeros(group * N, dtype=complex)
    S_all = np.zeros(group * N, dtype=complex)
    for g in range(group):
        dPHI_temp_start = g * max(conv)
        dPHI[g * N + non_zero_conv] = dPHI_temp[dPHI_temp_start + dPHI_temp_conv]
        S_all[g * N + non_zero_conv] = S[dPHI_temp_start + dPHI_temp_conv]
        for n in range(N):
            if conv[n] == 0:
                dPHI[g*N+n] = np.nan
                S_all[g*N+n] = np.nan
    dPHI_reshaped_plot = np.reshape(dPHI, (group, K_max, J_max, I_max))
    S_all_reshaped = np.reshape(S_all, (group, N))

#### 03. INVERT
    solver_type = 'noise'
    os.makedirs(f'{output_dir}/{case_name}_03_INVERT', exist_ok=True)
    output_INVERT = f'{output_dir}/{case_name}_03_INVERT/{case_name}'
    # --------------- MAP DETECTOR -------------------
    map_detector_reshaped = np.reshape(map_detector, (K_max, J_max, I_max))
    image_files_mag = []
    for k in range(K_max):
        filename_mag_map_detector = plot_heatmap_3D_general(map_detector_reshaped[k, :, :], 1, k+1, x, y, cmap='viridis', title=f'2D Plot of map_detector, Z={k+1}', output=output_INVERT, varname='map_detector', case_name=case_name, process_data='magnitude')
        image_files_mag.append(filename_mag_map_detector)
    gif_filename_mag_map_detector = f'{output_INVERT}_magnitude_map_detector_animation.gif'
    images_mag_map_detector = [Image.open(img) for img in image_files_mag]
    images_mag_map_detector[0].save(gif_filename_mag_map_detector, save_all=True, append_images=images_mag_map_detector[1:], duration=300, loop=0)

    map_detector_conv = np.zeros((group * max_conv))
    for n in range(N):
        if conv[n] != 0:
            map_detector_conv[conv[n] - 1] = map_detector[n]

    # --------------- MANIPULATE dPHI -------------------
    # Define zeroed dPHI as dPHI_zero
    dPHI_meas = np.zeros((group* N), dtype=complex) # 1D list, size (group * N)

    for g in range(group):
        dPHI_temp_start = g * max(conv)
        dPHI_meas[g * N + non_zero_conv] = dPHI_temp_meas[dPHI_temp_start + dPHI_temp_conv]
        for n in range(N):
            if conv[n] == 0:
                dPHI_meas[g*N+n] = np.nan

    # Plot dPHI_zero_reshaped
    dPHI_meas_reshaped = np.reshape(dPHI_meas, (group, K_max, J_max, I_max)) #3D array, size (group, J_max, I_max)
    for g in range(group):
        image_files_mag = []
        image_files_phase = []
        for k in range(K_max):
            filename_mag_dPHI_meas = plot_heatmap_3D_general(dPHI_meas_reshaped[g, k, :, :], g+1, k+1, x, y, cmap='viridis', title=f'2D Plot of dPHI{g+1}_meas, Z={k+1}', output=output_INVERT, varname='dPHI_meas', case_name=case_name, process_data='magnitude')
            filename_phase_dPHI_meas = plot_heatmap_3D_general(dPHI_meas_reshaped[g, k, :, :], g+1, k+1, x, y, cmap='viridis', title=f'2D Plot of dPHI{g+1}_meas, Z={k+1}', output=output_INVERT, varname='dPHI_meas', case_name=case_name, process_data='phase')
            image_files_mag.append(filename_mag_dPHI_meas)
            image_files_phase.append(filename_phase_dPHI_meas)
        gif_filename_mag_dPHI_meas = f'{output_INVERT}_magnitude_dPHI_meas_animation_G{g+1}.gif'
        gif_filename_phase_dPHI_meas = f'{output_INVERT}_phase_dPHI_meas_animation_G{g+1}.gif'
        images_mag_dPHI_meas = [Image.open(img) for img in image_files_mag]
        images_phase_dPHI_meas = [Image.open(img) for img in image_files_phase]
        images_mag_dPHI_meas[0].save(gif_filename_mag_dPHI_meas, save_all=True, append_images=images_mag_dPHI_meas[1:], duration=300, loop=0)
        images_phase_dPHI_meas[0].save(gif_filename_phase_dPHI_meas, save_all=True, append_images=images_phase_dPHI_meas[1:], duration=300, loop=0)

    # --------------- INTERPOLATE dPHI -------------------
    # Interpolation variables
    known_coords = np.array([(k, j, i) for k in range(K_max) for j in range(J_max) for i in range(I_max)
                            if map_detector[k * J_max * I_max + j * I_max + i] == 1])
    zero_coords = np.array([(k, j, i) for k in range(K_max) for j in range(J_max) for i in range(I_max)
                            if map_detector[k * J_max * I_max + j * I_max + i] == 0])

    pairwise_distances = pdist(known_coords, metric='euclidean')
    avg_distance = np.mean(pairwise_distances)  # You can also use np.median or another method
    epsilon = avg_distance / 16  # Set epsilon as a fraction of the average distance

    # Create a copy to avoid modifying the original array
    dPHI_interp = interpolate_dPHI_rbf_3D_rect(dPHI_meas, group, K_max, J_max, I_max, conv, known_coords, zero_coords, epsilon, rbf_function='thin_plate_spline')
    for g in range(group):
        for n in range(N):
            if map_detector[n] == 1:
                dPHI_interp[g*N+n] = dPHI_meas[g*N+n]
    dPHI_interp_array = np.reshape(dPHI_interp, (group, K_max, J_max, I_max))

    # Plot dPHI_interp
    for g in range(group):
        image_files_mag = []
        image_files_phase = []
        for k in range(K_max):
            filename_mag_dPHI_interp = plot_heatmap_3D_general(dPHI_interp_array[g, k, :, :], g+1, k+1, x, y, cmap='viridis', title=f'2D Plot of dPHI{g+1}_interp, Z={k+1}', output=output_INVERT, varname='dPHI_interp', case_name=case_name, process_data='magnitude')
            filename_phase_dPHI_interp = plot_heatmap_3D_general(dPHI_interp_array[g, k, :, :], g+1, k+1, x, y, cmap='viridis', title=f'2D Plot of dPHI{g+1}_interp, Z={k+1}', output=output_INVERT, varname='dPHI_interp', case_name=case_name, process_data='phase')
            image_files_mag.append(filename_mag_dPHI_interp)
            image_files_phase.append(filename_phase_dPHI_interp)
        gif_filename_mag_dPHI_interp = f'{output_INVERT}_magnitude_dPHI_interp_animation_G{g+1}.gif'
        gif_filename_phase_dPHI_interp = f'{output_INVERT}_phase_dPHI_interp_animation_G{g+1}.gif'
        images_mag_dPHI_interp = [Image.open(img) for img in image_files_mag]
        images_phase_dPHI_interp = [Image.open(img) for img in image_files_phase]
        images_mag_dPHI_interp[0].save(gif_filename_mag_dPHI_interp, save_all=True, append_images=images_mag_dPHI_interp[1:], duration=300, loop=0)
        images_phase_dPHI_interp[0].save(gif_filename_phase_dPHI_interp, save_all=True, append_images=images_phase_dPHI_interp[1:], duration=300, loop=0)

    # Define dPHI_temp_interp
    dPHI_temp_interp = dPHI_temp.copy()
    for g in range(group):
        for n in range(N):
            if conv[n] != 0:
                dPHI_temp_interp[g * max_conv + (conv[n] -1)] = dPHI_interp[g * N + n]
    dPHI_temp_INVERT = dPHI_temp_interp

    # Postprocess to save dPHI_interp
    dPHI_interp_reshaped = np.reshape(dPHI_interp, (group, N))
    dPHI_interp_reshaped_plot = np.reshape(dPHI_interp, (group, K_max, J_max, I_max))

    # OUTPUT
    print(f'Generating JSON output')
    output = {}
    for g in range(group):
        dPHI_groupname = f'dPHI{g+1}'
        dPHI_list = [{"real": x.real, "imaginary": x.imag} for x in dPHI_interp_reshaped[g]]
        output[dPHI_groupname] = dPHI_list

    # Save data to JSON file
    with open(f'{output_dir}/{case_name}_03_INVERT/{case_name}_output.json', 'w') as json_file:
        json.dump(output, json_file, indent=4)

    diff_flx1 = np.abs((np.array(dPHI_interp_reshaped_plot[0]) - np.array(dPHI_reshaped_plot[0]))/(np.array(dPHI_reshaped_plot[0] + 1E-20))) * 100
    diff_flx2 = np.abs((np.array(dPHI_interp_reshaped_plot[1]) - np.array(dPHI_reshaped_plot[1]))/(np.array(dPHI_reshaped_plot[1] + 1E-20))) * 100
    diff_flx = [[diff_flx1], [diff_flx2]]
    diff_dPHI_interp_array = np.array(diff_flx)
    diff_dPHI_interp_reshaped = diff_dPHI_interp_array.reshape(group, K_max, J_max, I_max)

    for g in range(group):
        image_files_mag = []
        image_files_phase = []
        for k in range(K_max):
            filename_mag_diff_dPHI_interp = plot_heatmap_3D_general(diff_dPHI_interp_reshaped[g, k, :, :], g+1, k+1, x, y, cmap='viridis', title=f'2D Plot of diff_dPHI{g+1}_interp, Z={k+1}', output=output_INVERT, varname='diff_dPHI_interp', case_name=case_name, process_data='magnitude')
            filename_phase_diff_dPHI_interp = plot_heatmap_3D_general(diff_dPHI_interp_reshaped[g, k, :, :], g+1, k+1, x, y, cmap='viridis', title=f'2D Plot of diff_dPHI{g+1}_interp, Z={k+1}', output=output_INVERT, varname='diff_dPHI_interp', case_name=case_name, process_data='phase')
            image_files_mag.append(filename_mag_diff_dPHI_interp)
            image_files_phase.append(filename_phase_diff_dPHI_interp)
        gif_filename_mag_diff_dPHI_interp = f'{output_INVERT}_magnitude_diff_dPHI_interp_animation_G{g+1}.gif'
        gif_filename_phase_diff_dPHI_interp = f'{output_INVERT}_phase_diff_dPHI_interp_animation_G{g+1}.gif'
        images_mag_diff_dPHI_interp = [Image.open(img) for img in image_files_mag]
        images_phase_diff_dPHI_interp = [Image.open(img) for img in image_files_phase]
        images_mag_diff_dPHI_interp[0].save(gif_filename_mag_diff_dPHI_interp, save_all=True, append_images=images_mag_diff_dPHI_interp[1:], duration=300, loop=0)
        images_phase_diff_dPHI_interp[0].save(gif_filename_phase_diff_dPHI_interp, save_all=True, append_images=images_phase_diff_dPHI_interp[1:], duration=300, loop=0)

    # --------------- LOAD GREEN'S FUNCTION -------------------
    # Plot G_matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(G_matrix.real, cmap='viridis', interpolation='nearest', origin='lower')
    plt.colorbar(label='Magnitude of G_matrix_full')
    plt.xlabel('Index')
    plt.ylabel('Index')
    plt.gca().invert_yaxis()
    plt.title('Plot of the Magnitude of G_matrix_full')
    plt.savefig(f'{output_dir}/{case_name}_03_INVERT/{case_name}_G_matrix_meas.png')

    # --------------- INTERPOLATE GREEN'S FUNCTION -------------------
    # Delete G_matrix_full at unknown position (column-wise at specific row)
    G_matrix_meas = G_matrix.copy()
    for g in range(group):
        for n in range(max_conv):
            if map_detector_conv[n] == 0:
                G_matrix_meas[g * max_conv + n, :] = 0 # Zeroing a column instead of a row

    plt.figure(figsize=(8, 6))
    plt.imshow(G_matrix_meas.real, cmap='viridis', interpolation='nearest', origin='lower')
    plt.colorbar(label='Magnitude of G_matrix_meas')
    plt.xlabel('Index')
    plt.ylabel('Index')
    plt.gca().invert_yaxis()
    plt.title('Plot of the Magnitude of G_matrix_meas')
    plt.savefig(f'{output_dir}/{case_name}_03_INVERT/{case_name}_G_matrix_meas.png')

    # Interpolate rows of the Green's function
    G_matrix_interp = G_matrix_meas
    G_matrix_interp_cols = np.zeros((group * max_conv, group * max_conv), dtype=complex)
    for g in range(group):
        for n in range(max_conv):
            G_mat_interp_temp = G_matrix_interp[:, g * max_conv + n]  # Extract a row
            print(f'Interpolating G_mat_interp_temp group {g+1}, position {n+1}')
            G_mat_interp_cols = interpolate_dPHI_rbf_3D_rect(G_mat_interp_temp, group, K_max, J_max, I_max, conv, known_coords, zero_coords, epsilon, rbf_function='thin_plate_spline') # Perform interpolation on the column
            G_matrix_interp_cols[:, g * max_conv + n] = G_mat_interp_cols  # Assign back to the row

    plt.figure(figsize=(8, 6))
    plt.imshow(G_matrix_interp_cols.real, cmap='viridis', interpolation='nearest', origin='lower')
    plt.colorbar(label='Magnitude of G_matrix_interp_cols')
    plt.xlabel('Index')
    plt.ylabel('Index')
    plt.gca().invert_yaxis()
    plt.title('Plot of the Magnitude of G_matrix_interp_cols')
    plt.savefig(f'{output_dir}/{case_name}_03_INVERT/{case_name}_G_matrix_interp_cols.png')

    # --------------- UNFOLD GREEEN'S FUNCTION USING DIRECT METHOD -------------------
    print(f'Solve for dS using Direct Method')
    G_mat_interp_inverse = scipy.linalg.pinv(G_matrix_interp_cols)

    # Plot G_matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(G_mat_interp_inverse.real, cmap='viridis', interpolation='nearest', origin='lower')
    plt.colorbar(label='Magnitude of G_mat_interp_inverse')
    plt.xlabel('Index')
    plt.ylabel('Index')
    plt.gca().invert_yaxis()
    plt.title('Plot of the Magnitude of G_mat_interp_inverse')
    plt.savefig(f'{output_dir}/{case_name}_03_INVERT/{case_name}_G_mat_interp_inverse.png')

    # UNFOLD ALL INTERPOLATED
    dS_unfold_INVERT_temp = np.dot(G_mat_interp_inverse, dPHI_temp_interp)
    dS_unfold_INVERT = np.zeros((group* N), dtype=complex)

    # POSTPROCESS
    print(f'Postprocessing to appropriate dPHI')
    non_zero_conv = np.nonzero(conv)[0]
    dS_unfold_temp_indices = conv_array[non_zero_conv] - 1

    for g in range(group):
        dS_unfold_temp_start = g * max(conv)
        dS_unfold_INVERT[g * N + non_zero_conv] = dS_unfold_INVERT_temp[dS_unfold_temp_start + dS_unfold_temp_indices]
        for n in range(N):
            if conv[n] == 0:
                dS_unfold_INVERT[g*N+n] = np.nan

    dS_unfold_INVERT_reshaped = np.reshape(dS_unfold_INVERT,(group,N))
    dS_unfold_INVERT_reshaped_plot = np.reshape(dS_unfold_INVERT, (group, K_max, J_max, I_max))

    for g in range(group):
        image_files_mag = []
        image_files_phase = []
        for k in range(K_max):
            filename_mag_dS_unfold_INVERT = plot_heatmap_3D_general(dS_unfold_INVERT_reshaped_plot[g, k, :, :], g+1, k+1, x, y, cmap='viridis', title=f'2D Plot of dS{g+1}_unfold_INVERT, Z={k+1}', output=output_INVERT, varname='dS_unfold_INVERT', case_name=case_name, process_data='magnitude')
            filename_phase_dS_unfold_INVERT = plot_heatmap_3D_general(dS_unfold_INVERT_reshaped_plot[g, k, :, :], g+1, k+1, x, y, cmap='viridis', title=f'2D Plot of dS{g+1}_unfold_INVERT, Z={k+1}', output=output_INVERT, varname='dS_unfold_INVERT', case_name=case_name, process_data='phase')
            image_files_mag.append(filename_mag_dS_unfold_INVERT)
            image_files_phase.append(filename_phase_dS_unfold_INVERT)
        gif_filename_mag_dS_unfold_INVERT = f'{output_INVERT}_magnitude_dS_unfold_INVERT_animation_G{g+1}.gif'
        gif_filename_phase_dS_unfold_INVERT = f'{output_INVERT}_phase_dS_unfold_INVERT_animation_G{g+1}.gif'
        images_mag_dS_unfold_INVERT = [Image.open(img) for img in image_files_mag]
        images_phase_dS_unfold_INVERT = [Image.open(img) for img in image_files_phase]
        images_mag_dS_unfold_INVERT[0].save(gif_filename_mag_dS_unfold_INVERT, save_all=True, append_images=images_mag_dS_unfold_INVERT[1:], duration=300, loop=0)
        images_phase_dS_unfold_INVERT[0].save(gif_filename_phase_dS_unfold_INVERT, save_all=True, append_images=images_phase_dS_unfold_INVERT[1:], duration=300, loop=0)

    # OUTPUT
    print(f'Generating JSON output for dS')
    output_direct1 = {}
    for g in range(group):
        dS_unfold_direct_groupname = f'dS_unfold{g+1}'
        dS_unfold_direct_list = [{"real": x.real, "imaginary": x.imag} for x in dS_unfold_INVERT_reshaped[g]]
        output_direct1[dS_unfold_direct_groupname] = dS_unfold_direct_list

    # Save data to JSON file
    with open(f'{output_dir}/{case_name}_03_INVERT/{case_name}_dS_unfold_INVERT_output.json', 'w') as json_file:
        json.dump(output_direct1, json_file, indent=4)

    # Calculate error and compare
    diff_S1_INVERT = (np.abs(np.array(dS_unfold_INVERT_reshaped[0]) - np.array(S_all_reshaped[0])) / (np.abs(np.array(S_all_reshaped[0])) + 1E-20)) * 100
    diff_S2_INVERT = (np.abs(np.array(dS_unfold_INVERT_reshaped[1]) - np.array(S_all_reshaped[1])) / (np.abs(np.array(S_all_reshaped[0])) + 1E-20)) * 100
    diff_S_INVERT = [[diff_S1_INVERT], [diff_S2_INVERT]]
    diff_S_INVERT_array = np.array(diff_S_INVERT)
    diff_S_INVERT_reshaped = diff_S_INVERT_array.reshape(group, K_max, J_max, I_max)

    for g in range(group):
        image_files_mag = []
        image_files_phase = []
        for k in range(K_max):
            filename_mag_diff_S_INVERT = plot_heatmap_3D_general(diff_S_INVERT_reshaped[g, k, :, :], g+1, k+1, x, y, cmap='viridis', title=f'2D Plot of diff_S{g+1}_INVERT, Z={k+1}', output=output_INVERT, varname='diff_S_INVERT', case_name=case_name, process_data='magnitude')
            filename_phase_diff_S_INVERT = plot_heatmap_3D_general(diff_S_INVERT_reshaped[g, k, :, :], g+1, k+1, x, y, cmap='viridis', title=f'2D Plot of diff_S{g+1}_INVERT, Z={k+1}', output=output_INVERT, varname='diff_S_INVERT', case_name=case_name, process_data='phase')
            image_files_mag.append(filename_mag_diff_S_INVERT)
            image_files_phase.append(filename_phase_diff_S_INVERT)
        gif_filename_mag_diff_S_INVERT = f'{output_INVERT}_magnitude_diff_S_INVERT_animation_G{g+1}.gif'
        gif_filename_phase_diff_S_INVERT = f'{output_INVERT}_phase_diff_S_INVERT_animation_G{g+1}.gif'
        images_mag_diff_S_INVERT = [Image.open(img) for img in image_files_mag]
        images_phase_diff_S_INVERT = [Image.open(img) for img in image_files_phase]
        images_mag_diff_S_INVERT[0].save(gif_filename_mag_diff_S_INVERT, save_all=True, append_images=images_mag_diff_S_INVERT[1:], duration=300, loop=0)
        images_phase_diff_S_INVERT[0].save(gif_filename_phase_diff_S_INVERT, save_all=True, append_images=images_phase_diff_S_INVERT[1:], duration=300, loop=0)

    return dPHI_temp_INVERT, dS_unfold_INVERT_temp

def main_unfold_3D_rect_zone(dPHI_temp_meas, dPHI_temp, S, G_matrix, group, N, I_max, J_max, K_max, conv, map_detector, map_zone, output_dir, case_name, x, y, z):
    max_conv = max(conv)
    conv_array = np.array(conv)
    non_zero_conv = np.nonzero(conv)[0]
    dPHI_temp_conv = conv_array[non_zero_conv] - 1
    dPHI = np.zeros(group * N, dtype=complex)
    S_all = np.zeros(group * N, dtype=complex)
    for g in range(group):
        dPHI_temp_start = g * max(conv)
        dPHI[g * N + non_zero_conv] = dPHI_temp[dPHI_temp_start + dPHI_temp_conv]
        S_all[g * N + non_zero_conv] = S[dPHI_temp_start + dPHI_temp_conv]
        for n in range(N):
            if conv[n] == 0:
                dPHI[g*N+n] = np.nan
                S_all[g*N+n] = np.nan
    dPHI_reshaped_plot = np.reshape(dPHI, (group, K_max, J_max, I_max))
    S_all_reshaped = np.reshape(S_all, (group, N))

##### 04. ZONE
    os.makedirs(f'{output_dir}/{case_name}_04_ZONE', exist_ok=True)
    output_ZONE = f'{output_dir}/{case_name}_04_ZONE/{case_name}'

    map_detector_conv = np.zeros((group * max_conv))
    for n in range(N):
        if conv[n] != 0:
            map_detector_conv[conv[n] - 1] = map_detector[n]

    # Expand the map zone
    map_zone_plot = map_zone.copy()
    for n in range(N):
        if conv[n] == 0:
            map_zone_plot[n] = np.nan
    map_zone_plot_reshaped = np.reshape(map_zone_plot, (K_max, J_max, I_max))
    image_files_mag = []
    for k in range(K_max):
        filename_mag_map_zone = plot_heatmap_3D_general(map_zone_plot_reshaped[k, :, :], 1, k+1, x, y, cmap='viridis', title=f'2D Plot of map_zone, Z={k+1}', output=output_ZONE, varname='map_zone', case_name=case_name, process_data='magnitude')
        image_files_mag.append(filename_mag_map_zone)
    gif_filename_mag_map_zone = f'{output_ZONE}_magnitude_map_zone_animation.gif'
    images_mag_map_zone = [Image.open(img) for img in image_files_mag]
    images_mag_map_zone[0].save(gif_filename_mag_map_zone, save_all=True, append_images=images_mag_map_zone[1:], duration=300, loop=0)

    map_zone_conv = np.zeros((group * max_conv))
    for n in range(N):
        if conv[n] != 0:
            map_zone_conv[conv[n] - 1] = map_zone[n]

    zone_length = np.zeros(int(max(map_zone_conv)))
    for z in range(int(max(map_zone_conv))):
        for n in range(max_conv):
            if map_zone_conv[n] == z + 1:
                zone_length[z] += 1

    # --------------- DIVIDE dPHI TO ZONES -------------------
    dPHI_temp_meas_zone = np.zeros((int(max(map_zone_conv)), group * max_conv), dtype=complex)
    for g in range(group):
        for n in range(max_conv):
            zone_index = int(map_zone_conv[n] - 1)
            dPHI_temp_meas_zone[zone_index][g * max_conv + n] = dPHI_temp_meas[g * max_conv + n]

    filename = f"{output_dir}/{case_name}_04_ZONE/{case_name}_dPHI_temp_meas_zone.txt"
    with open(filename, "w") as f:
        for zone_index, zone_data in enumerate(dPHI_temp_meas_zone):
            f.write(f"Zone {zone_index + 1}:\n")
            for value in zone_data:
                f.write(f"{value.real:.6e}+{value.imag:.6e}j \n")
            f.write("\n\n")  # Add a blank line between zones

    # --------------- MANIPULATE GREEN'S FUNCTION -------------------
    # Plot G_matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(G_matrix.real, cmap='viridis', interpolation='nearest', origin='lower')
    plt.colorbar(label='Magnitude of G_matrix_full')
    plt.xlabel('Index')
    plt.ylabel('Index')
    plt.gca().invert_yaxis()
    plt.title('Plot of the Magnitude of G_matrix_full')
    plt.savefig(f'{output_dir}/{case_name}_04_ZONE/{case_name}_G_matrix_full.png')

    # Delete G_matrix_full at unknown position (column-wise at specific row)
    G_matrix_meas = G_matrix.copy()
    for g in range(group):
        for n in range(max_conv):
            if map_detector_conv[n] == 0:
                G_matrix_meas[g * max_conv + n, :] = 0 # Zeroing a column instead of a row

    plt.figure(figsize=(8, 6))
    plt.imshow(G_matrix_meas.real, cmap='viridis', interpolation='nearest', origin='lower')
    plt.colorbar(label='Magnitude of G_matrix_meas')
    plt.xlabel('Index')
    plt.ylabel('Index')
    plt.gca().invert_yaxis()
    plt.title('Plot of the Magnitude of G_matrix_meas')
    plt.savefig(f'{output_dir}/{case_name}_04_ZONE/{case_name}_G_matrix_meas.png')

    ###################################################################################################
    # --------------- DIVIDE GREEN'S FUNCTION TO ZONES -------------------
    G_matrix_rows_zone = np.zeros((int(max(map_zone_conv)), group * max_conv, group * max_conv), dtype=complex)
    for g1 in range(group):
        for n1 in range(max_conv):
            for g2 in range(group):
                for n2 in range(max_conv):
                    zone_index = int(map_zone_conv[n2] - 1)
                    G_matrix_rows_zone[zone_index][g1 * max_conv + n1][g2 * max_conv + n2] = G_matrix_meas[g1 * max_conv + n1, g2 * max_conv + n2]

    for z in range(int(max(map_zone_conv))):
        plt.figure(figsize=(8, 6))
        plt.imshow(G_matrix_rows_zone[z].real, cmap='viridis', interpolation='nearest', origin='lower')
        plt.colorbar(label=f'Magnitude of G_matrix_rows_zone{z}')
        plt.xlabel('Index')
        plt.ylabel('Index')
        plt.gca().invert_yaxis()
        plt.title(f'Plot of the Magnitude of G_matrix_rows_zone{z}')
        plt.savefig(f'{output_dir}/{case_name}_04_ZONE/{case_name}_G_matrix_rows_zone{z}.png')

    ###################################################################################################
    # --------------- SOLVE FOR EACH ZONE -------------------
    dS_unfold_ZONE_temp = dPHI_temp_meas.copy()
    for g in range(group):
        for n in range(max_conv):
            dS_unfold_ZONE_temp[g * max_conv + n] = map_zone_conv[n]

    for z in range(int(max(map_zone_conv))):
        G_zone_matrix = G_matrix_rows_zone[z]
        non_zero_cols = ~np.all(G_zone_matrix == 0, axis=0)
        G_zone_mat = G_zone_matrix[:, non_zero_cols]

        plt.figure(figsize=(8, 6))
        plt.imshow(G_zone_mat.real, cmap='viridis', interpolation='nearest', origin='lower')
        plt.colorbar(label=f'Magnitude of G_matrix_zone{z}')
        plt.xlabel('Index')
        plt.ylabel('Index')
        plt.gca().invert_yaxis()
        plt.title(f'Plot of the Magnitude of G_matrix_zone{z}')
        plt.savefig(f'{output_dir}/{case_name}_04_ZONE/{case_name}_G_matrix_zone{z}.png')

        G_zone_square = []
        for g in range(group):
            for n in range(len(map_zone_conv)):
                if map_zone_conv[n] == z + 1:
                    G_zone_square.append(G_zone_mat[g * max_conv + n, :])

        plt.figure(figsize=(8, 6))
        plt.imshow(np.abs(G_zone_square), cmap='viridis', interpolation='nearest', origin='lower')
        plt.colorbar(label=f'Magnitude of G_zone_square_zone{z}')
        plt.xlabel('Index')
        plt.ylabel('Index')
        plt.gca().invert_yaxis()
        plt.title(f'Plot of the Magnitude of G_zone_square_zone{z}')
        plt.savefig(f'{output_dir}/{case_name}_04_ZONE/{case_name}_G_zone_square_zone{z}.png')

        # Extract the current zone noise flux
        zone_vector = dPHI_temp_meas_zone[z]
        zone_vector_new = []
        for g in range(group):
            for n in range(max_conv):
                if map_zone_conv[n] == z + 1:
                    zone_vector_new.append(zone_vector[n])

        zone_vector_new = np.array(zone_vector_new)

        # Inverse the G_matrix_rows_zone
        zone_matrix_inverse = scipy.linalg.pinv(G_zone_square)

        plt.figure(figsize=(8, 6))
        plt.imshow(zone_matrix_inverse.real, cmap='viridis', interpolation='nearest', origin='lower')
        plt.colorbar(label=f'Magnitude of G_inverse_zone{z}')
        plt.xlabel('Index')
        plt.ylabel('Index')
        plt.gca().invert_yaxis()
        plt.title(f'Plot of the Magnitude of G_inverse_zone{z}')
        plt.savefig(f'{output_dir}/{case_name}_04_ZONE/{case_name}_G_inverse_zone{z}.png')

        # Direct Solve
        dS_ZONE = []
        dS_ZONE = np.dot(zone_matrix_inverse, zone_vector_new)
        np.savetxt(f"{output_dir}/{case_name}_04_ZONE/{case_name}_dS_zone{z+1}.txt", dS_ZONE)
        print(f'Zone {z+1}: dS_zone length = {len(dS_ZONE)}')

        dS_ZONE_index = 0
        for i in range(len(dS_unfold_ZONE_temp)):
            if dS_unfold_ZONE_temp[i] == z+1:
                dS_unfold_ZONE_temp[i] = dS_ZONE[dS_ZONE_index]
                dS_ZONE_index += 1

        filename = f"{output_dir}/{case_name}_04_ZONE/{case_name}_dS_temp_meas_zone{z+1}.txt"
        with open(filename, "w") as f:
            for zone_index, zone_data in enumerate(dS_unfold_ZONE_temp):
                f.write(f"{zone_data.real:.6e}+{zone_data.imag:.6e}j \n")
            f.write("\n\n")  # Add a blank line between zones

    np.savetxt(f"{output_dir}/{case_name}_04_ZONE/{case_name}_dS_unfold_zone_temp.txt", dS_unfold_ZONE_temp)

    # POSTPROCESS
    dS_unfold_ZONE = np.zeros((group* N), dtype=complex)
    print(f'Postprocessing to appropriate dPHI')
    non_zero_conv = np.nonzero(conv)[0]
    dS_unfold_temp_indices = conv_array[non_zero_conv] - 1

    for g in range(group):
        dS_unfold_temp_start = g * max_conv

        for idx, non_zero_idx in enumerate(non_zero_conv):
            dS_unfold_ZONE[g * N + non_zero_idx] = dS_unfold_ZONE_temp[dS_unfold_temp_start + (conv_array[non_zero_idx] - 1)]    

        for n in range(N):
            if conv[n] == 0:
                dS_unfold_ZONE[g*N+n] = np.nan
    dS_unfold_ZONE_reshaped = np.reshape(dS_unfold_ZONE,(group,N))

    # Plot dPHI_sol_reshaped
    dS_unfold_ZONE_reshaped_plot = np.reshape(dS_unfold_ZONE, (group, K_max, J_max, I_max)) #3D array, size (group, J_max, I_max)
    for g in range(group):
        image_files_mag = []
        image_files_phase = []
        for k in range(K_max):
            filename_mag_dS_unfold_ZONE = plot_heatmap_3D_general(dS_unfold_ZONE_reshaped_plot[g, k, :, :], g+1, k+1, x, y, cmap='viridis', title=f'2D Plot of dS{g+1}_unfold_ZONE, Z={k+1}', output=output_ZONE, varname='dS_unfold_ZONE', case_name=case_name, process_data='magnitude')
            filename_phase_dS_unfold_ZONE = plot_heatmap_3D_general(dS_unfold_ZONE_reshaped_plot[g, k, :, :], g+1, k+1, x, y, cmap='viridis', title=f'2D Plot of dS{g+1}_unfold_ZONE, Z={k+1}', output=output_ZONE, varname='dS_unfold_ZONE', case_name=case_name, process_data='phase')
            image_files_mag.append(filename_mag_dS_unfold_ZONE)
            image_files_phase.append(filename_phase_dS_unfold_ZONE)
        gif_filename_mag_dS_unfold_ZONE = f'{output_ZONE}_magnitude_dS_unfold_ZONE_animation_G{g+1}.gif'
        gif_filename_phase_dS_unfold_ZONE = f'{output_ZONE}_phase_dS_unfold_ZONE_animation_G{g+1}.gif'
        images_mag_dS_unfold_ZONE = [Image.open(img) for img in image_files_mag]
        images_phase_dS_unfold_ZONE = [Image.open(img) for img in image_files_phase]
        images_mag_dS_unfold_ZONE[0].save(gif_filename_mag_dS_unfold_ZONE, save_all=True, append_images=images_mag_dS_unfold_ZONE[1:], duration=300, loop=0)
        images_phase_dS_unfold_ZONE[0].save(gif_filename_phase_dS_unfold_ZONE, save_all=True, append_images=images_phase_dS_unfold_ZONE[1:], duration=300, loop=0)

    # OUTPUT
    print(f'Generating JSON output for dS')
    output = {}
    for g in range(group):
        dS_unfold_ZONE_groupname = f'dS_unfold{g+1}'
        dS_unfold_ZONE_list = [{"real": x.real, "imaginary": x.imag} for x in dS_unfold_ZONE_reshaped[g]]
        output[dS_unfold_ZONE_groupname] = dS_unfold_ZONE_list

    # Save data to JSON file
    with open(f'{output_dir}/{case_name}_04_ZONE/{case_name}_dS_unfold_ZONE_output.json', 'w') as json_file:
        json.dump(output, json_file, indent=4)

    # Calculate error and compare
    diff_S1_ZONE = (np.abs(np.array(dS_unfold_ZONE_reshaped[0]) - np.array(S_all_reshaped[0])) / (np.abs(np.array(S_all_reshaped[0])) + 1E-20)) * 100
    diff_S2_ZONE = (np.abs(np.array(dS_unfold_ZONE_reshaped[1]) - np.array(S_all_reshaped[1])) / (np.abs(np.array(S_all_reshaped[0])) + 1E-20)) * 100
    diff_S_ZONE = [[diff_S1_ZONE], [diff_S2_ZONE]]
    diff_S_ZONE_array = np.array(diff_S_ZONE)
    diff_S_ZONE_reshaped = diff_S_ZONE_array.reshape(group, K_max, J_max, I_max)

    for g in range(group):
        image_files_mag = []
        image_files_phase = []
        for k in range(K_max):
            filename_mag_diff_S_ZONE = plot_heatmap_3D_general(diff_S_ZONE_reshaped[g, k, :, :], g+1, k+1, x, y, cmap='viridis', title=f'2D Plot of diff_S{g+1}_ZONE, Z={k+1}', output=output_ZONE, varname='diff_S_ZONE', case_name=case_name, process_data='magnitude')
            filename_phase_diff_S_ZONE = plot_heatmap_3D_general(diff_S_ZONE_reshaped[g, k, :, :], g+1, k+1, x, y, cmap='viridis', title=f'2D Plot of diff_S{g+1}_ZONE, Z={k+1}', output=output_ZONE, varname='diff_S_ZONE', case_name=case_name, process_data='phase')
            image_files_mag.append(filename_mag_diff_S_ZONE)
            image_files_phase.append(filename_phase_diff_S_ZONE)
        gif_filename_mag_diff_S_ZONE = f'{output_ZONE}_magnitude_diff_S_ZONE_animation_G{g+1}.gif'
        gif_filename_phase_diff_S_ZONE = f'{output_ZONE}_phase_diff_S_ZONE_animation_G{g+1}.gif'
        images_mag_diff_S_ZONE = [Image.open(img) for img in image_files_mag]
        images_phase_diff_S_ZONE = [Image.open(img) for img in image_files_phase]
        images_mag_diff_S_ZONE[0].save(gif_filename_mag_diff_S_ZONE, save_all=True, append_images=images_mag_diff_S_ZONE[1:], duration=300, loop=0)
        images_phase_diff_S_ZONE[0].save(gif_filename_phase_diff_S_ZONE, save_all=True, append_images=images_phase_diff_S_ZONE[1:], duration=300, loop=0)

    return dS_unfold_ZONE_temp

def main_unfold_3D_rect_scan(dPHI_temp_meas, dPHI_temp, S, G_matrix, group, N, I_max, J_max, K_max, conv, map_detector, map_zone, output_dir, case_name, x, y, z):
    max_conv = max(conv)
    conv_array = np.array(conv)
    non_zero_conv = np.nonzero(conv)[0]
    dPHI_temp_conv = conv_array[non_zero_conv] - 1
    dPHI = np.zeros(group * N, dtype=complex)
    S_all = np.zeros(group * N, dtype=complex)
    for g in range(group):
        dPHI_temp_start = g * max(conv)
        dPHI[g * N + non_zero_conv] = dPHI_temp[dPHI_temp_start + dPHI_temp_conv]
        S_all[g * N + non_zero_conv] = S[dPHI_temp_start + dPHI_temp_conv]
        for n in range(N):
            if conv[n] == 0:
                dPHI[g*N+n] = np.nan
                S_all[g*N+n] = np.nan
    dPHI_reshaped_plot = np.reshape(dPHI, (group, K_max, J_max, I_max))
    S_all_reshaped = np.reshape(S_all, (group, N))

##### 05. SCAN
    os.makedirs(f'{output_dir}/{case_name}_05_SCAN', exist_ok=True)
    output_SCAN = f'{output_dir}/{case_name}_05_SCAN/{case_name}'

    # Create tuple of detector pairs
    flux_pos = [index for index, value in enumerate(map_detector) if value == 1]

    flux_pos_conv = []
    for i, val in enumerate(flux_pos):
        flux_pos_conv.append(conv[val])
    detector_pair = list(combinations(flux_pos_conv, 2))
    
    delta_all = []
    for g in range(group):
        for n in range(max_conv):
            m = g * max_conv + n
            delta_AB = 0
            for p in range(len(detector_pair)):
                det_A = detector_pair[p][0] - 1
                det_B = detector_pair[p][1] - 1

                # Retrieve values for detectors A and B
                dPHI_A = dPHI_temp_meas[g * max_conv + (det_A)]
                dPHI_B = dPHI_temp_meas[g * max_conv + (det_B)]
                G_A = G_matrix[g * max_conv + (det_A)][m]
                G_B = G_matrix[g * max_conv + (det_B)][m]

                delta_AB += np.abs((dPHI_A / dPHI_B) - (G_A / G_B))

            delta_all.append(delta_AB)
            print(f'Done for group {g+1}, position {n}')

    # Save delta_all to a text file
    with open(f'{output_dir}/{case_name}_05_SCAN/{case_name}_delta_all.txt', 'w') as f:
        for item in delta_all:
            f.write(f"{item}\n")

    # Flatten dPHI_sol_temp_groups to a 1D list
    delta_all_full = np.zeros((group* N), dtype=complex) # 1D list, size (group * N)
    non_zero_conv = np.nonzero(conv)[0]

    # Create a copy for contour plotting before introducing np.nan
    for g in range(group):
        dPHI_temp_start = g * max_conv

        for idx, non_zero_idx in enumerate(non_zero_conv):
            delta_all_full[g * N + non_zero_idx] = delta_all[dPHI_temp_start + (conv_array[non_zero_idx] - 1)]

    # Now assign np.nan where necessary
    for g in range(group):
        for n in range(N):
            if conv[n] == 0:
                delta_all_full[g * N + n] = np.nan

    # Continue with other plotting routines
    delta_all_full_plot = np.reshape(delta_all_full, (group, K_max, J_max, I_max))  # 3D array, size (group, J_max, I_max)
    for g in range(group):
        image_files_mag = []
        for k in range(K_max):
            filename_mag_delta_all = plot_heatmap_3D_general(delta_all_full_plot[g, k, :, :], g+1, k+1, x, y, cmap='viridis', title=f'2D Plot of delta_AB{g+1} Magnitude, Z={k+1}', output=output_SCAN, varname=f'delta_AB{g+1}', case_name=case_name, process_data='magnitude')
            image_files_mag.append(filename_mag_delta_all)
        gif_filename_mag_delta_all = f'{output_SCAN}_magnitude_delta_all_animation.gif'
        images_mag_delta_all = [Image.open(img) for img in image_files_mag]
        images_mag_delta_all[0].save(gif_filename_mag_delta_all, save_all=True, append_images=images_mag_delta_all[1:], duration=300, loop=0)

    # Find minimum value and index
    min_value = min(delta_all)
    min_index = delta_all.index(min_value)
    for g in range(group):
        for n in range(N):
            if  g * max_conv + (conv[n] - 1) == min_index:
                print(f"Minimum value is {min_value} at index {min_index} (J = {n // I_max}, I = {n % I_max}) within group {g+1}")

    # Determine the scaling
    detector_loc = []
    for n in range(N):
        if map_detector[n] == 1:
            detector_loc.append(conv[n]-1)

    # Determine the scaling 
    G_sol_mat_temp_new = G_matrix[detector_loc[0]][min_index]
    dPHI_temp_meas_new = dPHI_temp_meas[detector_loc[0]]

    W = dPHI_temp_meas_new/G_sol_mat_temp_new #np.abs(dPHI_temp_meas_new/G_sol_mat_temp_new)
    print(f'magnitude of dS unfold is {W}')

    dS_unfold_SCAN_temp = [0.0] * group * max_conv
    dS_unfold_SCAN_temp[min_index] = W

    # Flatten dPHI_sol_temp_groups to a 1D list
    dS_unfold_SCAN = np.zeros((group* N), dtype=complex) # 1D list, size (group * N)
    non_zero_conv = np.nonzero(conv)[0]

    for g in range(group):
        dPHI_temp_start = g * max_conv

        for idx, non_zero_idx in enumerate(non_zero_conv):
            dS_unfold_SCAN[g * N + non_zero_idx] = dS_unfold_SCAN_temp[dPHI_temp_start + (conv_array[non_zero_idx] - 1)]    

        for n in range(N):
            if conv[n] == 0:
                dS_unfold_SCAN[g*N+n] = np.nan

    # Plot dPHI_sol_reshaped
    dS_unfold_SCAN_reshaped_plot = np.reshape(dS_unfold_SCAN, (group, K_max, J_max, I_max)) #3D array, size (group, J_max, I_max)
    dS_unfold_SCAN_reshaped = np.reshape(dS_unfold_SCAN,(group,N))
    for g in range(group):
        image_files_mag = []
        image_files_phase = []
        for k in range(K_max):
            filename_mag_dS_unfold_SCAN = plot_heatmap_3D_general(dS_unfold_SCAN_reshaped_plot[g, k, :, :], g+1, k+1, x, y, cmap='viridis', title=f'2D Plot of dS{g+1}_unfold_SCAN, Z={k+1}', output=output_SCAN, varname='dS_unfold_SCAN', case_name=case_name, process_data='magnitude')
            filename_phase_dS_unfold_SCAN = plot_heatmap_3D_general(dS_unfold_SCAN_reshaped_plot[g, k, :, :], g+1, k+1, x, y, cmap='viridis', title=f'2D Plot of dS{g+1}_unfold_SCAN, Z={k+1}', output=output_SCAN, varname='dS_unfold_SCAN', case_name=case_name, process_data='phase')
            image_files_mag.append(filename_mag_dS_unfold_SCAN)
            image_files_phase.append(filename_phase_dS_unfold_SCAN)
        gif_filename_mag_dS_unfold_SCAN = f'{output_SCAN}_magnitude_dS_unfold_SCAN_animation_G{g+1}.gif'
        gif_filename_phase_dS_unfold_SCAN = f'{output_SCAN}_phase_dS_unfold_SCAN_animation_G{g+1}.gif'
        images_mag_dS_unfold_SCAN = [Image.open(img) for img in image_files_mag]
        images_phase_dS_unfold_SCAN = [Image.open(img) for img in image_files_phase]
        images_mag_dS_unfold_SCAN[0].save(gif_filename_mag_dS_unfold_SCAN, save_all=True, append_images=images_mag_dS_unfold_SCAN[1:], duration=300, loop=0)
        images_phase_dS_unfold_SCAN[0].save(gif_filename_phase_dS_unfold_SCAN, save_all=True, append_images=images_phase_dS_unfold_SCAN[1:], duration=300, loop=0)

    # OUTPUT
    print(f'Generating JSON output for dS')
    output = {}
    for g in range(group):
        dS_unfold_SCAN_groupname = f'dS_unfold{g+1}'
        dS_unfold_SCAN_list = [{"real": x.real, "imaginary": x.imag} for x in dS_unfold_SCAN_reshaped[g]]
        output[dS_unfold_SCAN_groupname] = dS_unfold_SCAN_list

    # Save data to JSON file
    with open(f'{output_dir}/{case_name}_05_SCAN/{case_name}_dS_SCAN_output.json', 'w') as json_file:
        json.dump(output, json_file, indent=4)

    # Calculate error and compare
    diff_S1_SCAN = np.abs(np.array(dS_unfold_SCAN_reshaped[0]) - np.array(S_all_reshaped[0]))/(np.abs(np.array(S_all_reshaped[0])) + 1E-10) * 100
    diff_S2_SCAN = np.abs(np.array(dS_unfold_SCAN_reshaped[1]) - np.array(S_all_reshaped[1]))/(np.abs(np.array(S_all_reshaped[1])) + 1E-10) * 100
    diff_S_SCAN = [[diff_S1_SCAN], [diff_S2_SCAN]]
    diff_S_SCAN_array = np.array(diff_S_SCAN)
    diff_S_SCAN_reshaped = diff_S_SCAN_array.reshape(group, K_max, J_max, I_max)

    for g in range(group):
        image_files_mag = []
        image_files_phase = []
        for k in range(K_max):
            filename_mag_diff_S_SCAN = plot_heatmap_3D_general(diff_S_SCAN_reshaped[g, k, :, :], g+1, k+1, x, y, cmap='viridis', title=f'2D Plot of diff_S{g+1}_SCAN, Z={k+1}', output=output_SCAN, varname='diff_S_SCAN', case_name=case_name, process_data='magnitude')
            filename_phase_diff_S_SCAN = plot_heatmap_3D_general(diff_S_SCAN_reshaped[g, k, :, :], g+1, k+1, x, y, cmap='viridis', title=f'2D Plot of diff_S{g+1}_SCAN, Z={k+1}', output=output_SCAN, varname='diff_S_SCAN', case_name=case_name, process_data='phase')
            image_files_mag.append(filename_mag_diff_S_SCAN)
            image_files_phase.append(filename_phase_diff_S_SCAN)
        gif_filename_mag_diff_S_SCAN = f'{output_SCAN}_magnitude_diff_S_SCAN_animation_G{g+1}.gif'
        gif_filename_phase_diff_S_SCAN = f'{output_SCAN}_phase_diff_S_SCAN_animation_G{g+1}.gif'
        images_mag_diff_S_SCAN = [Image.open(img) for img in image_files_mag]
        images_phase_diff_S_SCAN = [Image.open(img) for img in image_files_phase]
        images_mag_diff_S_SCAN[0].save(gif_filename_mag_diff_S_SCAN, save_all=True, append_images=images_mag_diff_S_SCAN[1:], duration=300, loop=0)
        images_phase_diff_S_SCAN[0].save(gif_filename_phase_diff_S_SCAN, save_all=True, append_images=images_phase_diff_S_SCAN[1:], duration=300, loop=0)

    return dS_unfold_SCAN_temp

#######################################################################################################
def main_unfold_3D_rect_brute(dPHI_temp_meas, dPHI_temp, S, G_matrix, group, N, I_max, J_max, K_max, conv, map_detector, output_dir, case_name, x, y, z):
    max_conv = max(conv)
    conv_array = np.array(conv)
    non_zero_conv = np.nonzero(conv)[0]
    dPHI_temp_conv = conv_array[non_zero_conv] - 1
    dPHI = np.zeros(group * N, dtype=complex)
    S_all = np.zeros(group * N, dtype=complex)
    for g in range(group):
        dPHI_temp_start = g * max(conv)
        dPHI[g * N + non_zero_conv] = dPHI_temp[dPHI_temp_start + dPHI_temp_conv]
        S_all[g * N + non_zero_conv] = S[dPHI_temp_start + dPHI_temp_conv]
        for n in range(N):
            if conv[n] == 0:
                dPHI[g*N+n] = np.nan
                S_all[g*N+n] = np.nan
    dPHI_reshaped_plot = np.reshape(dPHI, (group, K_max, J_max, I_max))
    S_all_reshaped = np.reshape(S_all, (group, N))

#### 06. BRUTE FORCE
    os.makedirs(f'{output_dir}/{case_name}_06_BRUTE', exist_ok=True)
    output_BRUTE = f'{output_dir}/{case_name}_06_BRUTE/{case_name}'

    dPHI_temp_meas_reshaped = np.reshape(dPHI_temp_meas, (group, max_conv))

    # Define zeroed dPHI as dPHI_zero
    non_zero_conv = np.nonzero(conv)[0]
    dPHI_temp_conv = conv_array[non_zero_conv] - 1
    dPHI_meas = np.zeros((group* N), dtype=complex) # 1D list, size (group * N)

    for g in range(group):
        dPHI_temp_start = g * max(conv)
        dPHI_meas[g * N + non_zero_conv] = dPHI_temp_meas[dPHI_temp_start + dPHI_temp_conv]
        for n in range(N):
            if conv[n] == 0:
                dPHI_meas[g*N+n] = np.nan

    # Plot dPHI_zero_reshaped
    dPHI_meas_reshaped = np.reshape(dPHI_meas, (group, K_max, J_max, I_max)) #3D array, size (group, J_max, I_max)
    for g in range(group):
        image_files_mag = []
        image_files_phase = []
        for k in range(K_max):
            filename_mag_dPHI_meas = plot_heatmap_3D_general(dPHI_meas_reshaped[g, k, :, :], g+1, k+1, x, y, cmap='viridis', title=f'2D Plot of dPHI{g+1}_meas, Z={k+1}', output=output_BRUTE, varname='dPHI_meas', case_name=case_name, process_data='magnitude')
            filename_phase_dPHI_meas = plot_heatmap_3D_general(dPHI_meas_reshaped[g, k, :, :], g+1, k+1, x, y, cmap='viridis', title=f'2D Plot of dPHI{g+1}_meas, Z={k+1}', output=output_BRUTE, varname='dPHI_meas', case_name=case_name, process_data='phase')
            image_files_mag.append(filename_mag_dPHI_meas)
            image_files_phase.append(filename_phase_dPHI_meas)
        gif_filename_mag_dPHI_meas = f'{output_BRUTE}_magnitude_dPHI_meas_animation_G{g+1}.gif'
        gif_filename_phase_dPHI_meas = f'{output_BRUTE}_phase_dPHI_meas_animation_G{g+1}.gif'
        images_mag_dPHI_meas = [Image.open(img) for img in image_files_mag]
        images_phase_dPHI_meas = [Image.open(img) for img in image_files_phase]
        images_mag_dPHI_meas[0].save(gif_filename_mag_dPHI_meas, save_all=True, append_images=images_mag_dPHI_meas[1:], duration=300, loop=0)
        images_phase_dPHI_meas[0].save(gif_filename_phase_dPHI_meas, save_all=True, append_images=images_phase_dPHI_meas[1:], duration=300, loop=0)

    non_zero_indices = np.nonzero(dPHI_temp_meas)[0]
    dPHI_temp_meas = np.array(dPHI_temp_meas)

    # Create dictionary to store each atom (column of G_matrix_full)
    G_dictionary = {}
    for g in range(group):
        for n in range(max_conv):
            m = g * max_conv + n
            G_dictionary[f"G_g{g+1}_n{n+1}"] = G_matrix[:, m]

    # Sample the dictionary atoms at known points (sparse form)
    G_dictionary_sampled = {k: np.zeros_like(dPHI_temp_meas, dtype=complex) for k in G_dictionary}
    for k, o in G_dictionary.items():
        G_dictionary_sampled[k][non_zero_indices] = o[non_zero_indices]

    # Initialize variables
    valid_solution_BRUTE = False  # Flag to indicate a valid solution
    tol_BRUTE = 1E-8

    # Brute force over all combinations of atoms
    atom_keys = list(G_dictionary_sampled.keys())
    num_atoms = len(atom_keys)
    residual_file = f"{output_dir}/{case_name}_06_BRUTE/{case_name}_subset_residuals.txt"
    chunk_size = 10000

    # Check if the file exists, if yes, delete it
    if os.path.exists(residual_file):
        os.remove(residual_file)
        print(f"Existing file '{residual_file}' deleted.")

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
                A = np.array([G_dictionary_sampled[k] for k in subset]).T #np.column_stack([G_dictionary_sampled[k] for k in subset])
                coeffs = np.linalg.lstsq(A, dPHI_temp_meas, rcond=None)[0]
                coefficients = dict(zip(subset, coeffs))
                residual = dPHI_temp_meas - A @ coeffs
                residual_norm = np.linalg.norm(residual)

                # Append to file without changing loop structure
                with open(residual_file, "a") as file:
                    file.write(f"{subset}, {residual_norm:.6e}\n")

                # Check if residual norm meets tolerance
                if residual_norm < tol_BRUTE:
                    print(f'Subsets {subset} pass the residual tolerance')
                    valid_solution_BRUTE = True  # Criterion satisfied
                    print(f"Valid solution found with number of sources = {num_source} and atoms = {subset}.")
                    coefficients = dict(zip(subset, coeffs))
                    dPHI_temp_BRUTE = sum(c * G_dictionary[k] for k, c in coefficients.items())
                    break  # Exit the outer loop

                if valid_solution_BRUTE:
                    break  # Exit the subset loop

                iter_BRUTE += 1

            if valid_solution_BRUTE:
                break  # Exit the outer loop

        if valid_solution_BRUTE:
            break  # Exit the outer loop

    if not valid_solution_BRUTE:
        print("No valid solution found with brute force.")

    ###################################################################################################
    if valid_solution_BRUTE:
        # Reshape reconstructed signal
        non_zero_conv = np.nonzero(conv)[0]
        dPHI_temp_conv = conv_array[non_zero_conv] - 1
        dPHI_BRUTE = np.zeros((group* N), dtype=complex) # 1D list, size (group * N)

        for g in range(group):
            dPHI_temp_start = g * max(conv)
            dPHI_BRUTE[g * N + non_zero_conv] = dPHI_temp_BRUTE[dPHI_temp_start + dPHI_temp_conv]
            for n in range(N):
                if conv[n] == 0:
                    dPHI_BRUTE[g*N+n] = np.nan

        # Plot dPHI_zero_reshaped
        dPHI_BRUTE_reshaped = np.reshape(dPHI_BRUTE, (group, K_max, J_max, I_max)) #3D array, size (group, J_max, I_max)
        for g in range(group):
            image_files_mag = []
            image_files_phase = []
            for k in range(K_max):
                filename_mag_dPHI_BRUTE = plot_heatmap_3D_general(dPHI_BRUTE_reshaped[g, k, :, :], g+1, k+1, x, y, cmap='viridis', title=f'2D Plot of dPHI{g+1}_BRUTE, Z={k+1}', output=output_BRUTE, varname='dPHI_BRUTE', case_name=case_name, process_data='magnitude')
                filename_phase_dPHI_BRUTE = plot_heatmap_3D_general(dPHI_BRUTE_reshaped[g, k, :, :], g+1, k+1, x, y, cmap='viridis', title=f'2D Plot of dPHI{g+1}_BRUTE, Z={k+1}', output=output_BRUTE, varname='dPHI_BRUTE', case_name=case_name, process_data='phase')
                image_files_mag.append(filename_mag_dPHI_BRUTE)
                image_files_phase.append(filename_phase_dPHI_BRUTE)
            gif_filename_mag_dPHI_BRUTE = f'{output_BRUTE}_magnitude_dPHI_BRUTE_animation_G{g+1}.gif'
            gif_filename_phase_dPHI_BRUTE = f'{output_BRUTE}_phase_dPHI_BRUTE_animation_G{g+1}.gif'
            images_mag_dPHI_BRUTE = [Image.open(img) for img in image_files_mag]
            images_phase_dPHI_BRUTE = [Image.open(img) for img in image_files_phase]
            images_mag_dPHI_BRUTE[0].save(gif_filename_mag_dPHI_BRUTE, save_all=True, append_images=images_mag_dPHI_BRUTE[1:], duration=300, loop=0)
            images_phase_dPHI_BRUTE[0].save(gif_filename_phase_dPHI_BRUTE, save_all=True, append_images=images_phase_dPHI_BRUTE[1:], duration=300, loop=0)

        ######################################################################################################
        # --------------- UNFOLD GREEEN'S FUNCTION USING DIRECT METHOD -------------------
        print(f'Solve for dS using Direct Method')
        G_inverse = scipy.linalg.inv(G_matrix)

        # Plot G_matrix
        plt.figure(figsize=(8, 6))
        plt.imshow(G_inverse.real, cmap='viridis', interpolation='nearest', origin='lower')
        plt.colorbar(label='Magnitude of G_inverse')
        plt.xlabel('Index')
        plt.ylabel('Index')
        plt.gca().invert_yaxis()
        plt.title('Plot of the Magnitude of G_inverse')
        plt.savefig(f'{output_dir}/{case_name}_06_BRUTE/{case_name}_G_inverse.png')

        # UNFOLD ALL INTERPOLATED
        dS_unfold_BRUTE_temp = np.dot(G_inverse, dPHI_temp_BRUTE)

        # POSTPROCESS
        print(f'Postprocessing to appropriate dPHI')
        non_zero_conv = np.nonzero(conv)[0]
        dS_unfold_temp_indices = conv_array[non_zero_conv] - 1
        dS_unfold_BRUTE = np.zeros((group* N), dtype=complex)

        for g in range(group):
            dS_unfold_temp_start = g * max(conv)
            dS_unfold_BRUTE[g * N + non_zero_conv] = dS_unfold_BRUTE_temp[dS_unfold_temp_start + dS_unfold_temp_indices]
            for n in range(N):
                if conv[n] == 0:
                    dS_unfold_BRUTE[g*N+n] = np.nan

        dS_unfold_BRUTE_reshaped = np.reshape(dS_unfold_BRUTE,(group,N))
        dS_unfold_BRUTE_plot = np.reshape(dS_unfold_BRUTE, (group, K_max, J_max, I_max))

        for g in range(group):
            image_files_mag = []
            image_files_phase = []
            for k in range(K_max):
                filename_mag_dS_unfold_BRUTE = plot_heatmap_3D_general(dS_unfold_BRUTE_plot[g, k, :, :], g+1, k+1, x, y, cmap='viridis', title=f'2D Plot of dS{g+1}_unfold_BRUTE, Z={k+1}', output=output_BRUTE, varname='dS_unfold_BRUTE', case_name=case_name, process_data='magnitude')
                filename_phase_dS_unfold_BRUTE = plot_heatmap_3D_general(dS_unfold_BRUTE_plot[g, k, :, :], g+1, k+1, x, y, cmap='viridis', title=f'2D Plot of dS{g+1}_unfold_BRUTE, Z={k+1}', output=output_BRUTE, varname='dS_unfold_BRUTE', case_name=case_name, process_data='phase')
                image_files_mag.append(filename_mag_dS_unfold_BRUTE)
                image_files_phase.append(filename_phase_dS_unfold_BRUTE)
            gif_filename_mag_dS_unfold_BRUTE = f'{output_BRUTE}_magnitude_dS_unfold_BRUTE_animation_G{g+1}.gif'
            gif_filename_phase_dS_unfold_BRUTE = f'{output_BRUTE}_phase_dS_unfold_BRUTE_animation_G{g+1}.gif'
            images_mag_dS_unfold_BRUTE = [Image.open(img) for img in image_files_mag]
            images_phase_dS_unfold_BRUTE = [Image.open(img) for img in image_files_phase]
            images_mag_dS_unfold_BRUTE[0].save(gif_filename_mag_dS_unfold_BRUTE, save_all=True, append_images=images_mag_dS_unfold_BRUTE[1:], duration=300, loop=0)
            images_phase_dS_unfold_BRUTE[0].save(gif_filename_phase_dS_unfold_BRUTE, save_all=True, append_images=images_phase_dS_unfold_BRUTE[1:], duration=300, loop=0)

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

        # Calculate error and compare
        diff_S1_BRUTE = np.abs(np.array(dS_unfold_BRUTE_reshaped[0]) - np.array(S_all_reshaped[0]))/(np.abs(np.array(S_all_reshaped[0])) + 1E-10) * 100
        diff_S2_BRUTE = np.abs(np.array(dS_unfold_BRUTE_reshaped[1]) - np.array(S_all_reshaped[1]))/(np.abs(np.array(S_all_reshaped[1])) + 1E-10) * 100
        diff_S_BRUTE = [[diff_S1_BRUTE], [diff_S2_BRUTE]]
        diff_S_BRUTE_array = np.array(diff_S_BRUTE)
        diff_S_BRUTE_reshaped = diff_S_BRUTE_array.reshape(group, K_max, J_max, I_max)

        for g in range(group):
            image_files_mag = []
            image_files_phase = []
            for k in range(K_max):
                filename_mag_diff_S_BRUTE = plot_heatmap_3D_general(diff_S_BRUTE_reshaped[g, k, :, :], g+1, k+1, x, y, cmap='viridis', title=f'2D Plot of diff_S{g+1}_BRUTE, Z={k+1}', output=output_BRUTE, varname='diff_S_BRUTE', case_name=case_name, process_data='magnitude')
                filename_phase_diff_S_BRUTE = plot_heatmap_3D_general(diff_S_BRUTE_reshaped[g, k, :, :], g+1, k+1, x, y, cmap='viridis', title=f'2D Plot of diff_S{g+1}_BRUTE, Z={k+1}', output=output_BRUTE, varname='diff_S_BRUTE', case_name=case_name, process_data='phase')
                image_files_mag.append(filename_mag_diff_S_BRUTE)
                image_files_phase.append(filename_phase_diff_S_BRUTE)
            gif_filename_mag_diff_S_BRUTE = f'{output_BRUTE}_magnitude_diff_S_BRUTE_animation_G{g+1}.gif'
            gif_filename_phase_diff_S_BRUTE = f'{output_BRUTE}_phase_diff_S_BRUTE_animation_G{g+1}.gif'
            images_mag_diff_S_BRUTE = [Image.open(img) for img in image_files_mag]
            images_phase_diff_S_BRUTE = [Image.open(img) for img in image_files_phase]
            images_mag_diff_S_BRUTE[0].save(gif_filename_mag_diff_S_BRUTE, save_all=True, append_images=images_mag_diff_S_BRUTE[1:], duration=300, loop=0)
            images_phase_diff_S_BRUTE[0].save(gif_filename_phase_diff_S_BRUTE, save_all=True, append_images=images_phase_diff_S_BRUTE[1:], duration=300, loop=0)

    return dPHI_temp_BRUTE, dS_unfold_BRUTE_temp

#######################################################################################################
def main_unfold_3D_rect_back(dPHI_temp_meas, dPHI_temp, S, G_matrix, group, N, I_max, J_max, K_max, conv, output_dir, case_name, x, y, z):
    max_conv = max(conv)
    conv_array = np.array(conv)
    non_zero_conv = np.nonzero(conv)[0]
    dPHI_temp_conv = conv_array[non_zero_conv] - 1
    dPHI = np.zeros(group * N, dtype=complex)
    S_all = np.zeros(group * N, dtype=complex)
    for g in range(group):
        dPHI_temp_start = g * max(conv)
        dPHI[g * N + non_zero_conv] = dPHI_temp[dPHI_temp_start + dPHI_temp_conv]
        S_all[g * N + non_zero_conv] = S[dPHI_temp_start + dPHI_temp_conv]
        for n in range(N):
            if conv[n] == 0:
                dPHI[g*N+n] = np.nan
                S_all[g*N+n] = np.nan
    dPHI_reshaped_plot = np.reshape(dPHI, (group, K_max, J_max, I_max))
    S_all_reshaped = np.reshape(S_all, (group, N))

##### 07. BACKWARD ELIMINATION
    os.makedirs(f'{output_dir}/{case_name}_07_BACK', exist_ok=True)
    output_BACK = f'{output_dir}/{case_name}_07_BACK/{case_name}'

    dPHI_temp_meas_reshaped = np.reshape(dPHI_temp_meas, (group, max_conv))

    # Define zeroed dPHI as dPHI_zero
    non_zero_conv = np.nonzero(conv)[0]
    dPHI_temp_conv = conv_array[non_zero_conv] - 1
    dPHI_meas = np.zeros((group* N), dtype=complex) # 1D list, size (group * N)

    for g in range(group):
        dPHI_temp_start = g * max(conv)
        dPHI_meas[g * N + non_zero_conv] = dPHI_temp_meas[dPHI_temp_start + dPHI_temp_conv]
        for n in range(N):
            if conv[n] == 0:
                dPHI_meas[g*N+n] = np.nan

    # Plot dPHI_zero_reshaped
    dPHI_meas_reshaped = np.reshape(dPHI_meas, (group, K_max, J_max, I_max)) #3D array, size (group, J_max, I_max)
    for g in range(group):
        image_files_mag = []
        image_files_phase = []
        for k in range(K_max):
            filename_mag_dPHI_meas = plot_heatmap_3D_general(dPHI_meas_reshaped[g, k, :, :], g+1, k+1, x, y, cmap='viridis', title=f'2D Plot of dPHI{g+1}_meas, Z={k+1}', output=output_BACK, varname='dPHI_meas', case_name=case_name, process_data='magnitude')
            filename_phase_dPHI_meas = plot_heatmap_3D_general(dPHI_meas_reshaped[g, k, :, :], g+1, k+1, x, y, cmap='viridis', title=f'2D Plot of dPHI{g+1}_meas, Z={k+1}', output=output_BACK, varname='dPHI_meas', case_name=case_name, process_data='phase')
            image_files_mag.append(filename_mag_dPHI_meas)
            image_files_phase.append(filename_phase_dPHI_meas)
        gif_filename_mag_dPHI_meas = f'{output_BACK}_magnitude_dPHI_meas_animation_G{g+1}.gif'
        gif_filename_phase_dPHI_meas = f'{output_BACK}_phase_dPHI_meas_animation_G{g+1}.gif'
        images_mag_dPHI_meas = [Image.open(img) for img in image_files_mag]
        images_phase_dPHI_meas = [Image.open(img) for img in image_files_phase]
        images_mag_dPHI_meas[0].save(gif_filename_mag_dPHI_meas, save_all=True, append_images=images_mag_dPHI_meas[1:], duration=300, loop=0)
        images_phase_dPHI_meas[0].save(gif_filename_phase_dPHI_meas, save_all=True, append_images=images_phase_dPHI_meas[1:], duration=300, loop=0)

    non_zero_indices = np.nonzero(dPHI_temp_meas)[0]
    dPHI_temp_meas = np.array(dPHI_temp_meas)

    # Create dictionary to store each atom (column of G_matrix_full)
    G_dictionary = {}
    for g in range(group):
        for n in range(max_conv):
            m = g * max_conv + n
            G_dictionary[f"G_g{g+1}_n{n+1}"] = G_matrix[:, m]

    # Sample the dictionary atoms at known points (sparse form)
    G_dictionary_sampled = {k: np.zeros_like(dPHI_temp_meas, dtype=complex) for k in G_dictionary}
    for k, o in G_dictionary.items():
        G_dictionary_sampled[k][non_zero_indices] = o[non_zero_indices]

    # Initialize variables for the higher loop
    valid_solution_BACK = False  # Flag to indicate a valid solution
    tol_BACK = 1E-10
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
            coeffs = np.linalg.lstsq(A, dPHI_temp_meas, rcond=None)[0]
            coefficients = dict(zip(selected_atoms, coeffs))
            residual = dPHI_temp_meas - A @ coeffs
            residual_norm = np.linalg.norm(residual)
            # Compute contributions (absolute value of coefficients)
            contributions = {atom: abs(coeff) / max(abs(coeffs)) for atom, coeff in zip(selected_atoms, coeffs)}

            # Validate the reconstructed signal against the criterion
            if residual_norm < tol_BACK:
                valid_solution = True  # Criterion satisfied
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
                    print(f'Criteria not met using Backward Elimination.')
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
        coeffs = np.linalg.lstsq(A, dPHI_temp_meas, rcond=None)[0]
        coefficients = dict(zip(valid_solution_BACK, coeffs))
        dPHI_temp_BACK = sum(c * G_dictionary[k] for k, c in coefficients.items())
    else:
        print("Failed to find a valid solution within the maximum number of outer iterations.")

    ###################################################################################################
    if valid_solution_BACK:
        # Reshape reconstructed signal
        non_zero_conv = np.nonzero(conv)[0]
        dPHI_temp_conv = conv_array[non_zero_conv] - 1
        dPHI_BACK = np.zeros((group* N), dtype=complex) # 1D list, size (group * N)

        for g in range(group):
            dPHI_temp_start = g * max(conv)
            dPHI_BACK[g * N + non_zero_conv] = dPHI_temp_BACK[dPHI_temp_start + dPHI_temp_conv]
            for n in range(N):
                if conv[n] == 0:
                    dPHI_BACK[g*N+n] = np.nan

        # Plot dPHI_zero_reshaped
        dPHI_BACK_reshaped = np.reshape(dPHI_BACK, (group, K_max, J_max, I_max)) #3D array, size (group, J_max, I_max)
        for g in range(group):
            image_files_mag = []
            image_files_phase = []
            for k in range(K_max):
                filename_mag_dPHI_BACK = plot_heatmap_3D_general(dPHI_BACK_reshaped[g, k, :, :], g+1, k+1, x, y, cmap='viridis', title=f'2D Plot of dPHI{g+1}_BACK, Z={k+1}', output=output_BACK, varname='dPHI_BACK', case_name=case_name, process_data='magnitude')
                filename_phase_dPHI_BACK = plot_heatmap_3D_general(dPHI_BACK_reshaped[g, k, :, :], g+1, k+1, x, y, cmap='viridis', title=f'2D Plot of dPHI{g+1}_BACK, Z={k+1}', output=output_BACK, varname='dPHI_BACK', case_name=case_name, process_data='phase')
                image_files_mag.append(filename_mag_dPHI_BACK)
                image_files_phase.append(filename_phase_dPHI_BACK)
            gif_filename_mag_dPHI_BACK = f'{output_BACK}_magnitude_dPHI_BACK_animation_G{g+1}.gif'
            gif_filename_phase_dPHI_BACK = f'{output_BACK}_phase_dPHI_BACK_animation_G{g+1}.gif'
            images_mag_dPHI_BACK = [Image.open(img) for img in image_files_mag]
            images_phase_dPHI_BACK = [Image.open(img) for img in image_files_phase]
            images_mag_dPHI_BACK[0].save(gif_filename_mag_dPHI_BACK, save_all=True, append_images=images_mag_dPHI_BACK[1:], duration=300, loop=0)
            images_phase_dPHI_BACK[0].save(gif_filename_phase_dPHI_BACK, save_all=True, append_images=images_phase_dPHI_BACK[1:], duration=300, loop=0)

        ######################################################################################################
        # --------------- UNFOLD GREEEN'S FUNCTION USING DIRECT METHOD -------------------
        print(f'Solve for dS using Direct Method')
        G_inverse = scipy.linalg.inv(G_matrix)

        # Plot G_matrix
        plt.figure(figsize=(8, 6))
        plt.imshow(G_inverse.real, cmap='viridis', interpolation='nearest', origin='lower')
        plt.colorbar(label='Magnitude of G_inverse')
        plt.xlabel('Index')
        plt.ylabel('Index')
        plt.gca().invert_yaxis()
        plt.title('Plot of the Magnitude of G_inverse')
        plt.savefig(f'{output_dir}/{case_name}_07_BACK/{case_name}_G_inverse.png')

        # UNFOLD ALL INTERPOLATED
        dS_unfold_BACK_temp = np.dot(G_inverse, dPHI_temp_BACK)

        # POSTPROCESS
        print(f'Postprocessing to appropriate dPHI')
        non_zero_conv = np.nonzero(conv)[0]
        dS_unfold_temp_indices = conv_array[non_zero_conv] - 1
        dS_unfold_BACK = np.zeros((group* N), dtype=complex)

        for g in range(group):
            dS_unfold_temp_start = g * max(conv)
            dS_unfold_BACK[g * N + non_zero_conv] = dS_unfold_BACK_temp[dS_unfold_temp_start + dS_unfold_temp_indices]
            for n in range(N):
                if conv[n] == 0:
                    dS_unfold_BACK[g*N+n] = np.nan

        dS_unfold_BACK_reshaped = np.reshape(dS_unfold_BACK,(group,N))
        dS_unfold_BACK_plot = np.reshape(dS_unfold_BACK, (group, K_max, J_max, I_max))

        for g in range(group):
            image_files_mag = []
            image_files_phase = []
            for k in range(K_max):
                filename_mag_dS_unfold_BACK = plot_heatmap_3D_general(dS_unfold_BACK_plot[g, k, :, :], g+1, k+1, x, y, cmap='viridis', title=f'2D Plot of dS{g+1}_unfold_BACK, Z={k+1}', output=output_BACK, varname='dS_unfold_BACK', case_name=case_name, process_data='magnitude')
                filename_phase_dS_unfold_BACK = plot_heatmap_3D_general(dS_unfold_BACK_plot[g, k, :, :], g+1, k+1, x, y, cmap='viridis', title=f'2D Plot of dS{g+1}_unfold_BACK, Z={k+1}', output=output_BACK, varname='dS_unfold_BACK', case_name=case_name, process_data='phase')
                image_files_mag.append(filename_mag_dS_unfold_BACK)
                image_files_phase.append(filename_phase_dS_unfold_BACK)
            gif_filename_mag_dS_unfold_BACK = f'{output_BACK}_magnitude_dS_unfold_BACK_animation_G{g+1}.gif'
            gif_filename_phase_dS_unfold_BACK = f'{output_BACK}_phase_dS_unfold_BACK_animation_G{g+1}.gif'
            images_mag_dS_unfold_BACK = [Image.open(img) for img in image_files_mag]
            images_phase_dS_unfold_BACK = [Image.open(img) for img in image_files_phase]
            images_mag_dS_unfold_BACK[0].save(gif_filename_mag_dS_unfold_BACK, save_all=True, append_images=images_mag_dS_unfold_BACK[1:], duration=300, loop=0)
            images_phase_dS_unfold_BACK[0].save(gif_filename_phase_dS_unfold_BACK, save_all=True, append_images=images_phase_dS_unfold_BACK[1:], duration=300, loop=0)

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

        # Calculate error and compare
        diff_S1_BACK = np.abs(np.array(dS_unfold_BACK_reshaped[0]) - np.array(S_all_reshaped[0]))/(np.abs(np.array(S_all_reshaped[0])) + 1E-7) * 100
        diff_S2_BACK = np.abs(np.array(dS_unfold_BACK_reshaped[1]) - np.array(S_all_reshaped[1]))/(np.abs(np.array(S_all_reshaped[1])) + 1E-7) * 100
        diff_S_BACK = [[diff_S1_BACK], [diff_S2_BACK]]
        diff_S_BACK_array = np.array(diff_S_BACK)
        diff_S_BACK_reshaped = diff_S_BACK_array.reshape(group, J_max, I_max)

        for g in range(group):
            image_files_mag = []
            image_files_phase = []
            for k in range(K_max):
                filename_mag_diff_S_BACK = plot_heatmap_3D_general(diff_S_BACK_reshaped[g, k, :, :], g+1, k+1, x, y, cmap='viridis', title=f'2D Plot of diff_S{g+1}_BACK, Z={k+1}', output=output_BACK, varname='diff_S_BACK', case_name=case_name, process_data='magnitude')
                filename_phase_diff_S_BACK = plot_heatmap_3D_general(diff_S_BACK_reshaped[g, k, :, :], g+1, k+1, x, y, cmap='viridis', title=f'2D Plot of diff_S{g+1}_BACK, Z={k+1}', output=output_BACK, varname='diff_S_BACK', case_name=case_name, process_data='phase')
                image_files_mag.append(filename_mag_diff_S_BACK)
                image_files_phase.append(filename_phase_diff_S_BACK)
            gif_filename_mag_diff_S_BACK = f'{output_BACK}_magnitude_diff_S_BACK_animation_G{g+1}.gif'
            gif_filename_phase_diff_S_BACK = f'{output_BACK}_phase_diff_S_BACK_animation_G{g+1}.gif'
            images_mag_diff_S_BACK = [Image.open(img) for img in image_files_mag]
            images_phase_diff_S_BACK = [Image.open(img) for img in image_files_phase]
            images_mag_diff_S_BACK[0].save(gif_filename_mag_diff_S_BACK, save_all=True, append_images=images_mag_diff_S_BACK[1:], duration=300, loop=0)
            images_phase_diff_S_BACK[0].save(gif_filename_phase_diff_S_BACK, save_all=True, append_images=images_phase_diff_S_BACK[1:], duration=300, loop=0)

    else:
        print("No valid solution found with backward elimination.")

    return dPHI_temp_BACK, dS_unfold_BACK_temp

def main_unfold_3D_rect_greedy_optimized(dPHI_temp_meas, dPHI_temp, S, G_matrix, group, N, I_max, J_max, K_max, conv, output_dir, case_name, x, y, z):
    conv_array = np.array(conv)
    max_conv = max(conv)
    non_zero_conv = np.nonzero(conv)[0]
    dPHI_temp_conv = conv_array[non_zero_conv] - 1
    dPHI = np.zeros(group * N, dtype=complex)
    S_all = np.zeros(group * N, dtype=complex)
    for g in range(group):
        dPHI_temp_start = g * max(conv)
        dPHI[g * N + non_zero_conv] = dPHI_temp[dPHI_temp_start + dPHI_temp_conv]
        S_all[g * N + non_zero_conv] = S[dPHI_temp_start + dPHI_temp_conv]
        for n in range(N):
            if conv[n] == 0:
                dPHI[g*N+n] = np.nan
                S_all[g*N+n] = np.nan
    dPHI_reshaped_plot = np.reshape(dPHI, (group, K_max, J_max, I_max))
    S_all_reshaped = np.reshape(S_all, (group, N))

    non_zero_indices = np.nonzero(dPHI_temp_meas)[0]
    dPHI_temp_meas = np.array(dPHI_temp_meas)

    # Create dictionary to store each atom (column of G_matrix_full)
    G_dictionary = {}
    for g in range(group):
        for n in range(max_conv):
            m = g * max_conv + n
            G_dictionary[f"G_g{g+1}_n{n+1}"] = G_matrix[:, m]

    # Sample the dictionary atoms at known points (sparse form)
    G_dictionary_sampled = {k: np.zeros_like(dPHI_temp_meas, dtype=complex) for k in G_dictionary}
    for k, o in G_dictionary.items():
        G_dictionary_sampled[k][non_zero_indices] = o[non_zero_indices]

    keys = list(G_dictionary_sampled.keys())
    key_to_idx = {k: i for i, k in enumerate(keys)}
    G_full = np.column_stack([G_dictionary[k] for k in keys]).astype(complex, copy=False)#  (full, non‑sampled dictionary)
    X_full = np.column_stack([G_dictionary_sampled[k] for k in keys]).astype(complex, copy=False)
    y_full = np.asarray(dPHI_temp_meas, dtype=complex)   
    obs = np.asarray(non_zero_indices)
    X = X_full[obs, :]          # shape (M_obs, N)
    y_dPHI = y_full[obs]             # shape (M_obs,)
    num_atoms = X.shape[1]      # N

##### 08. GREEDY
    os.makedirs(f'{output_dir}/{case_name}_08_GREEDY', exist_ok=True)
    output_GREEDY = f'{output_dir}/{case_name}_08_GREEDY/{case_name}'

    dPHI_temp_meas_reshaped = np.reshape(dPHI_temp_meas, (group, max_conv))

    # Define zeroed dPHI as dPHI_zero
    non_zero_conv = np.nonzero(conv)[0]
    dPHI_temp_conv = conv_array[non_zero_conv] - 1
    dPHI_meas = np.zeros((group* N), dtype=complex) # 1D list, size (group * N)

    for g in range(group):
        dPHI_temp_start = g * max(conv)
        dPHI_meas[g * N + non_zero_conv] = dPHI_temp_meas[dPHI_temp_start + dPHI_temp_conv]
        for n in range(N):
            if conv[n] == 0:
                dPHI_meas[g*N+n] = np.nan

    # Plot dPHI_zero_reshaped
    dPHI_meas_reshaped = np.reshape(dPHI_meas, (group, K_max, J_max, I_max)) #3D array, size (group, J_max, I_max)
    for g in range(group):
        image_files_mag = []
        image_files_phase = []
        for k in range(K_max):
            filename_mag_dPHI_meas = plot_heatmap_3D_general(dPHI_meas_reshaped[g, k, :, :], g+1, k+1, x, y, cmap='viridis', title=f'2D Plot of dPHI{g+1}_meas, Z={k+1}', output=output_GREEDY, varname='dPHI_meas', case_name=case_name, process_data='magnitude')
            filename_phase_dPHI_meas = plot_heatmap_3D_general(dPHI_meas_reshaped[g, k, :, :], g+1, k+1, x, y, cmap='viridis', title=f'2D Plot of dPHI{g+1}_meas, Z={k+1}', output=output_GREEDY, varname='dPHI_meas', case_name=case_name, process_data='phase')
            image_files_mag.append(filename_mag_dPHI_meas)
            image_files_phase.append(filename_phase_dPHI_meas)
        gif_filename_mag_dPHI_meas = f'{output_GREEDY}_magnitude_dPHI_meas_animation_G{g+1}.gif'
        gif_filename_phase_dPHI_meas = f'{output_GREEDY}_phase_dPHI_meas_animation_G{g+1}.gif'
        images_mag_dPHI_meas = [Image.open(img) for img in image_files_mag]
        images_phase_dPHI_meas = [Image.open(img) for img in image_files_phase]
        images_mag_dPHI_meas[0].save(gif_filename_mag_dPHI_meas, save_all=True, append_images=images_mag_dPHI_meas[1:], duration=300, loop=0)
        images_phase_dPHI_meas[0].save(gif_filename_phase_dPHI_meas, save_all=True, append_images=images_phase_dPHI_meas[1:], duration=300, loop=0)

    # Initialize variables for the higher loop
    valid_solution_GREEDY = False  # Flag to indicate a valid solution
    outer_iter = 0
#    inner_iter = 0
    tol_GREEDY = 1E-10  # Stopping tolerance
    comb_first_atom = 1
#    selected_atoms = []
    contribution_threshold = 1e-6  # Define the contribution threshold
#    all_outer_iter_len = len(G_dictionary) #+ len(list(combinations(G_dictionary_sampled.keys(), 2)))

    # Build all possible "first atom" combinations once
    first_atom_indices_iter = list(combinations(range(num_atoms), comb_first_atom))

    # Dictionary to store valid solutions with term counts
    valid_solutions_GREEDY = {}

    # Precompute column norms once outside all loops:
    col_norm_sq = np.einsum('ij,ij->j', X.conj(), X).real  # shape (N,)

#    # Iterate over possible first atoms
#    while outer_iter < all_outer_iter_len:
#        first_atom_iter = combinations(G_dictionary_sampled.keys(), comb_first_atom)
#
#        for first_atom in first_atom_iter:
#            # Initialize residual and coefficients
#            residual = dPHI_temp_meas.copy()
#            selected_atoms = list(first_atom)
#            coefficients = []
#            residual_norm = np.linalg.norm(residual)
#
#            # Form the initial matrix with the first atom
#            A = np.array([G_dictionary_sampled[k] for k in first_atom]).T
#            coeffs = np.linalg.lstsq(A, dPHI_temp_meas, rcond=None)[0]
#            residual = dPHI_temp_meas - A @ coeffs
#            residual_norm = np.linalg.norm(residual)
#            print(f"Outer iteration {outer_iter+1}: Trying first atom {first_atom}, current residual norm = {residual_norm:.6e}")
#
#            # Perform Greedy Residual Minimization
#            prev_selected_atoms_len = 0
#            constant_len_counter = 0
#            while residual_norm > tol_GREEDY:
#                residuals = {}
#                for k in combinations(G_dictionary_sampled.keys(), comb_first_atom):
#                    # Skip this combination if any atom is already selected
#                    if any(atom in selected_atoms for atom in k):
#                        continue
#                    temp_atoms = selected_atoms + list(k) #[k]
#                    A_temp = np.array([G_dictionary_sampled[a] for a in temp_atoms]).T
#                    coeffs_temp = np.linalg.lstsq(A_temp, dPHI_temp_meas, rcond=None)[0]
#                    residuals[k] = np.linalg.norm(dPHI_temp_meas - A_temp @ coeffs_temp)
#                chosen_atom = min(residuals, key=residuals.get)
#
#                if isinstance(chosen_atom, tuple):  # If chosen_atom is a tuple, extend the list
#                    for atom in chosen_atom:
#                        if atom not in selected_atoms:
#                            selected_atoms.append(atom)
#                else:  # If chosen_atom is a single key, append it
#                    if chosen_atom not in selected_atoms:
#                        selected_atoms.append(chosen_atom)
#
#                # Form matrix of selected atoms
#                A = np.array([G_dictionary_sampled[k] for k in selected_atoms]).T
#
#                # Solve least-squares problem to update coefficients
#                coeffs = np.linalg.lstsq(A, dPHI_temp_meas, rcond=None)[0]
#                coefficients = dict(zip(selected_atoms, coeffs))
#
#                # Update residual
#                residual = dPHI_temp_meas - A @ coeffs
#                residual_norm = np.linalg.norm(residual)
#
#                print(f'   Chosen atom = {chosen_atom}, length of selected atoms = {len(selected_atoms)}, current residual norm = {residual_norm:.6e}')
#
#                # Check if the length of selected_atoms remains constant
#                if len(selected_atoms) == prev_selected_atoms_len:
#                    constant_len_counter += 1
#                else:
#                    constant_len_counter = 0  # Reset counter if length changes
#
#                prev_selected_atoms_len = len(selected_atoms)
#
#                if constant_len_counter >= 10:
#                    print("   Terminating loop: Length of selected_atoms remained constant for 10 iterations.")
#                    break
#
#                inner_iter += 1
#
#            # Check for low contribution atoms
#            contributions = {atom: abs(coeff) / max(abs(coeffs)) for atom, coeff in zip(selected_atoms, coeffs)}
#            low_contribution_atoms = [atom for atom, contribution in contributions.items() if contribution < contribution_threshold]
#
#            if low_contribution_atoms:
#                for atom in low_contribution_atoms:
#                    if atom in selected_atoms:
#                        selected_atoms.remove(atom)
#            print(f"   Selected_atoms = {selected_atoms}, residual norm = {residual_norm:.6e}")
#
#            # Validate the reconstructed signal against the criterion
#            if residual_norm < tol_GREEDY:
#                valid_solution = True  # Criterion satisfied
#                print(f"Valid solution found with first atom {first_atom} in outer iteration {outer_iter+1}.")
#                valid_solutions_GREEDY[first_atom] = selected_atoms #len(selected_atoms)
#            else:
#                print(f"Criterion not met with first atom {first_atom}. Restarting with a new atom.")
#
#            outer_iter += 1

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
        print(f"The best valid solution is with atom {best_atom} with selected atoms = {valid_solutions_GREEDY[best_atom]}.")
        valid_solution_GREEDY = valid_solutions_GREEDY[best_atom]

        A = np.array([G_dictionary_sampled[k] for k in valid_solution_GREEDY]).T
        coeffs = np.linalg.lstsq(A, dPHI_temp_meas, rcond=None)[0]
        coefficients = dict(zip(valid_solution_GREEDY, coeffs))
        dPHI_temp_GREEDY = sum(c * G_dictionary[k] for k, c in zip(valid_solution_GREEDY, coeffs))
    else:
        print("Failed to find a valid solution within the maximum number of outer iterations.")

    ####################################################################################################
    if valid_solution_GREEDY:
        # Reshape reconstructed signal
        non_zero_conv = np.nonzero(conv)[0]
        dPHI_temp_conv = conv_array[non_zero_conv] - 1
        dPHI_GREEDY = np.zeros((group* N), dtype=complex) # 1D list, size (group * N)

        for g in range(group):
            dPHI_temp_start = g * max(conv)
            dPHI_GREEDY[g * N + non_zero_conv] = dPHI_temp_GREEDY[dPHI_temp_start + dPHI_temp_conv]
            for n in range(N):
                if conv[n] == 0:
                    dPHI_GREEDY[g*N+n] = np.nan

        # Plot dPHI_zero_reshaped
        dPHI_GREEDY_reshaped = np.reshape(dPHI_GREEDY, (group, K_max, J_max, I_max)) #3D array, size (group, J_max, I_max)
        for g in range(group):
            image_files_mag = []
            image_files_phase = []
            for k in range(K_max):
                filename_mag_dPHI_GREEDY = plot_heatmap_3D_general(dPHI_GREEDY_reshaped[g, k, :, :], g+1, k+1, x, y, cmap='viridis', title=f'2D Plot of dPHI{g+1}_GREEDY, Z={k+1}', output=output_GREEDY, varname='dPHI_GREEDY', case_name=case_name, process_data='magnitude')
                filename_phase_dPHI_GREEDY = plot_heatmap_3D_general(dPHI_GREEDY_reshaped[g, k, :, :], g+1, k+1, x, y, cmap='viridis', title=f'2D Plot of dPHI{g+1}_GREEDY, Z={k+1}', output=output_GREEDY, varname='dPHI_GREEDY', case_name=case_name, process_data='phase')
                image_files_mag.append(filename_mag_dPHI_GREEDY)
                image_files_phase.append(filename_phase_dPHI_GREEDY)
            gif_filename_mag_dPHI_GREEDY = f'{output_GREEDY}_magnitude_dPHI_GREEDY_animation_G{g+1}.gif'
            gif_filename_phase_dPHI_GREEDY = f'{output_GREEDY}_phase_dPHI_GREEDY_animation_G{g+1}.gif'
            images_mag_dPHI_GREEDY = [Image.open(img) for img in image_files_mag]
            images_phase_dPHI_GREEDY = [Image.open(img) for img in image_files_phase]
            images_mag_dPHI_GREEDY[0].save(gif_filename_mag_dPHI_GREEDY, save_all=True, append_images=images_mag_dPHI_GREEDY[1:], duration=300, loop=0)
            images_phase_dPHI_GREEDY[0].save(gif_filename_phase_dPHI_GREEDY, save_all=True, append_images=images_phase_dPHI_GREEDY[1:], duration=300, loop=0)

        ######################################################################################################
        # --------------- UNFOLD GREEEN'S FUNCTION USING DIRECT METHOD -------------------
        print(f'\nSolve for dS using Direct Method')
#        G_inverse = scipy.linalg.inv(G_matrix)
#
#        # UNFOLD ALL INTERPOLATED
#        dS_unfold_GREEDY_temp = np.dot(G_inverse, dPHI_temp_GREEDY)
        lu, piv = lu_factor(G_matrix)
        dS_unfold_GREEDY_temp = lu_solve((lu, piv), dPHI_temp_GREEDY)
    
        # POSTPROCESS
        print(f'Postprocessing to appropriate dPHI')
        non_zero_conv = np.nonzero(conv)[0]
        dS_unfold_temp_indices = conv_array[non_zero_conv] - 1
        dS_unfold_GREEDY = np.zeros((group* N), dtype=complex)

        for g in range(group):
            dS_unfold_temp_start = g * max(conv)
            dS_unfold_GREEDY[g * N + non_zero_conv] = dS_unfold_GREEDY_temp[dS_unfold_temp_start + dS_unfold_temp_indices]
            for n in range(N):
                if conv[n] == 0:
                    dS_unfold_GREEDY[g*N+n] = np.nan

        dS_unfold_GREEDY_reshaped = np.reshape(dS_unfold_GREEDY,(group,N))
        dS_unfold_GREEDY_plot = np.reshape(dS_unfold_GREEDY, (group, K_max, J_max, I_max))

        for g in range(group):
            image_files_mag = []
            image_files_phase = []
            for k in range(K_max):
                filename_mag_dS_unfold_GREEDY = plot_heatmap_3D_general(dS_unfold_GREEDY_plot[g, k, :, :], g+1, k+1, x, y, cmap='viridis', title=f'2D Plot of dS{g+1}_unfold_GREEDY, Z={k+1}', output=output_GREEDY, varname='dS_unfold_GREEDY', case_name=case_name, process_data='magnitude')
                filename_phase_dS_unfold_GREEDY = plot_heatmap_3D_general(dS_unfold_GREEDY_plot[g, k, :, :], g+1, k+1, x, y, cmap='viridis', title=f'2D Plot of dS{g+1}_unfold_GREEDY, Z={k+1}', output=output_GREEDY, varname='dS_unfold_GREEDY', case_name=case_name, process_data='phase')
                image_files_mag.append(filename_mag_dS_unfold_GREEDY)
                image_files_phase.append(filename_phase_dS_unfold_GREEDY)
            gif_filename_mag_dS_unfold_GREEDY = f'{output_GREEDY}_magnitude_dS_unfold_GREEDY_animation_G{g+1}.gif'
            gif_filename_phase_dS_unfold_GREEDY = f'{output_GREEDY}_phase_dS_unfold_GREEDY_animation_G{g+1}.gif'
            images_mag_dS_unfold_GREEDY = [Image.open(img) for img in image_files_mag]
            images_phase_dS_unfold_GREEDY = [Image.open(img) for img in image_files_phase]
            images_mag_dS_unfold_GREEDY[0].save(gif_filename_mag_dS_unfold_GREEDY, save_all=True, append_images=images_mag_dS_unfold_GREEDY[1:], duration=300, loop=0)
            images_phase_dS_unfold_GREEDY[0].save(gif_filename_phase_dS_unfold_GREEDY, save_all=True, append_images=images_phase_dS_unfold_GREEDY[1:], duration=300, loop=0)

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

        # Calculate error and compare
        diff_S1_GREEDY = np.abs(np.array(dS_unfold_GREEDY_reshaped[0]) - np.array(S_all_reshaped[0]))/(np.abs(np.array(S_all_reshaped[0])) + 1E-6) * 100
        diff_S2_GREEDY = np.abs(np.array(dS_unfold_GREEDY_reshaped[1]) - np.array(S_all_reshaped[1]))/(np.abs(np.array(S_all_reshaped[1])) + 1E-6) * 100
        diff_S_GREEDY = [[diff_S1_GREEDY], [diff_S2_GREEDY]]
        diff_S_GREEDY_array = np.array(diff_S_GREEDY)
        diff_S_GREEDY_reshaped = diff_S_GREEDY_array.reshape(group, K_max, J_max, I_max)

        for g in range(group):
            image_files_mag = []
            image_files_phase = []
            for k in range(K_max):
                filename_mag_diff_S_GREEDY = plot_heatmap_3D_general(diff_S_GREEDY_reshaped[g, k, :, :], g+1, k+1, x, y, cmap='viridis', title=f'2D Plot of diff_S{g+1}_GREEDY, Z={k+1}', output=output_GREEDY, varname='diff_S_GREEDY', case_name=case_name, process_data='magnitude')
                filename_phase_diff_S_GREEDY = plot_heatmap_3D_general(diff_S_GREEDY_reshaped[g, k, :, :], g+1, k+1, x, y, cmap='viridis', title=f'2D Plot of diff_S{g+1}_GREEDY, Z={k+1}', output=output_GREEDY, varname='diff_S_GREEDY', case_name=case_name, process_data='phase')
                image_files_mag.append(filename_mag_diff_S_GREEDY)
                image_files_phase.append(filename_phase_diff_S_GREEDY)
            gif_filename_mag_diff_S_GREEDY = f'{output_GREEDY}_magnitude_diff_S_GREEDY_animation_G{g+1}.gif'
            gif_filename_phase_diff_S_GREEDY = f'{output_GREEDY}_phase_diff_S_GREEDY_animation_G{g+1}.gif'
            images_mag_diff_S_GREEDY = [Image.open(img) for img in image_files_mag]
            images_phase_diff_S_GREEDY = [Image.open(img) for img in image_files_phase]
            images_mag_diff_S_GREEDY[0].save(gif_filename_mag_diff_S_GREEDY, save_all=True, append_images=images_mag_diff_S_GREEDY[1:], duration=300, loop=0)
            images_phase_diff_S_GREEDY[0].save(gif_filename_phase_diff_S_GREEDY, save_all=True, append_images=images_phase_diff_S_GREEDY[1:], duration=300, loop=0)

    return dPHI_temp_GREEDY, dS_unfold_GREEDY_temp

