import numpy as np
import json
import time
import os
import sys
from scipy.integrate import trapezoid
from itertools import combinations
from petsc4py import PETSc
from scipy.linalg import lstsq

# Prevent .pyc file generation
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True

start_time = time.time()

from SRC.UTILS import Utils
from SRC.MATRIX_BUILDER import *
from SRC.METHODS import *
from SRC.POSTPROCESS import PostProcessor
from SRC.SOLVERFACTORY import SolverFactory
from SRC.SRC_UNFOLDING_2D_HEXX import *

#######################################################################################################
# INPUTS
original_sys_path = sys.path.copy()
sys.path.append('../')

#from INPUTS.OBJECTIVES45_TEST03_2DTriMG_HTTR_AVS import *
#from INPUTS.OBJECTIVES45_TEST04_2DTriMG_HTTR_FAV import *
from INPUTS.OBJECTIVES45_TEST10_2DTriMG_HTTR_AVS3S import *

# Restore the original sys.path
sys.path = original_sys_path

#######################################################################################################
def main():
    start_time = time.time()

    output_dir = f'../OUTPUTS/{input_name}/{case_name}_UNFOLDING'

##### Forward Simulation
    solver_type = 'forward'
    os.makedirs(f'{output_dir}/{case_name}_UNFOLDING/{case_name}_{solver_type.upper()}', exist_ok=True)
    conv_hexx = convert_2D_hexx(I_max, J_max, D)
    conv_tri, conv_hexx_ext = convert_2D_tri(I_max, J_max, conv_hexx, level)
    conv_tri_array = np.array(conv_tri)
    max_conv = max(conv_tri)
    conv_neighbor, tri_indices, x, y, all_triangles = calculate_neighbors_2D(s, I_max, J_max, conv_hexx, level)

    matrix_builder = MatrixBuilderForward2DHexx(group, I_max, J_max, conv_tri, conv_neighbor, TOT, SIGS_reshaped, BC, h, level, D, chi, NUFIS)
    M, F_FORWARD = matrix_builder.build_forward_matrices()

    solver = SolverFactory.get_solver_power2DHexx(solver_type, group, conv_tri, M, F_FORWARD, h, precond, tol=1E-10)
    keff, PHI_temp = solver.solve()
    PHI, PHI_reshaped, PHI_temp_reshaped = PostProcessor.postprocess_power2DHexx(PHI_temp, conv_tri, group, N_hexx)

    output = {"keff": keff.real}
    for g in range(len(PHI_reshaped)):
        phi_groupname = f'PHI{g + 1}_{solver_type.upper()}'
        output[phi_groupname] = [val.real for val in PHI_reshaped[g]]

    with open(f'{output_dir}/{case_name}_UNFOLDING/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_output.json', 'w') as json_file:
        json.dump(output, json_file, indent=4)

##### Adjoint Simulation
    solver_type = 'adjoint'
    os.makedirs(f'{output_dir}/{case_name}_UNFOLDING/{case_name}_{solver_type.upper()}', exist_ok=True)

    matrix_builder = MatrixBuilderAdjoint2DHexx(group, I_max, J_max, conv_tri, conv_neighbor, TOT, SIGS_reshaped, BC, h, level, D, chi, NUFIS)
    M, F_ADJOINT = matrix_builder.build_adjoint_matrices()

    solver = SolverFactory.get_solver_power2DHexx(solver_type, group, conv_tri, M, F_ADJOINT, h, precond, tol=1E-10)
    keff, PHI_ADJ_temp = solver.solve()
    PHI_ADJ, PHI_ADJ_reshaped, PHI_ADJ_temp_reshaped = PostProcessor.postprocess_power2DHexx(PHI_ADJ_temp, conv_tri, group, N_hexx)

    output = {"keff": keff.real}
    for g in range(len(PHI_ADJ_reshaped)):
        phi_groupname = f'PHI{g + 1}_{solver_type.upper()}'
        output[phi_groupname] = [val.real for val in PHI_ADJ_reshaped[g]]

    with open(f'{output_dir}/{case_name}_UNFOLDING/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_output.json', 'w') as json_file:
        json.dump(output, json_file, indent=4)

##### NOISE PREPARATION
    # Noise Input Manipulation
    dTOT_hexx = expand_XS_hexx_2D(group, J_max, I_max, dTOT, level)
    dSIGS_hexx = expand_SIGS_hexx_2D(group, J_max, I_max, dSIGS_reshaped, level)
    chi_hexx = expand_XS_hexx_2D(group, J_max, I_max, chi, level)
    dNUFIS_hexx = expand_XS_hexx_2D(group, J_max, I_max, dNUFIS, level)

    if noise_section == 1:
        # Collect all non-zero indices of dTOT_hexx for each group
        for g in range(group):
            for n in range(N_hexx):
                if dTOT_hexx[g][n] != 0:
                    noise_tri_index = n//(6 * (4 ** (level - 1))) * (6 * (4 ** (level - 1))) + 2
                    if n != noise_tri_index:
                        dTOT_hexx[g][n] = 0
    else:
        pass
    if type_noise == 'FVX' or type_noise == 'FAV':
        if level < 2:
            raise ValueError('Vibrating Assembly type noise only works if level at least 2')

    hex_centers, hex_vertices = generate_pointy_hex_grid(s, I_max, J_max)
    triangle_neighbors_global = find_triangle_neighbors_2D(all_triangles, precision=6)

    if type_noise == 'FXV':
        dTOT_hexx, dNUFIS_hexx = XS2D_FXV(level, group, J_max, I_max, dTOT, dNUFIS, fav_strength, diff_X_ABS, diff_X_NUFIS, all_triangles, hex_centers, triangle_neighbors_global)
    elif type_noise == 'FAV':
        dTOT_hexx, dNUFIS_hexx = XS2D_FAV(level, group, J_max, I_max, dTOT, dNUFIS, fav_strength, diff_X_ABS, diff_X_NUFIS, all_triangles, hex_centers, triangle_neighbors_global)

    # --------------- MAP DETECTOR -------------------
    # Expand the map detector
    p = 6 * (4 ** (level - 1))
    map_detector_hexx = [0] * I_max * J_max * p
    map_detector_temp = np.reshape(map_detector, (J_max, I_max))
    for j in range(J_max):
        for i in range(I_max):
            m = j * I_max + i
            for k in range(p):
                if k == 3:
                    map_detector_hexx[m * p + k] = map_detector_temp[j][i]

##### BASE (Noise, Green's function, and solve)
    dPHI_temp = main_unfold_2D_hexx_noise(PHI_temp, keff, group, I_max, J_max, N_hexx, conv_tri, conv_neighbor, TOT, SIGS_reshaped, BC, h, level, D, chi, NUFIS, v, Beff, omega, l, dTOT_hexx, dSIGS_hexx, dNUFIS_hexx, chi_hexx, noise_section, type_noise, map_detector_hexx, output_dir, case_name, precond, tri_indices, x, y)
    G_matrix = main_unfold_2D_hexx_green(keff, group, I_max, J_max, N_hexx, conv_tri, conv_neighbor, TOT, SIGS_reshaped, BC, h, level, D, chi, NUFIS, v, Beff, omega, l, dTOT_hexx, dSIGS_hexx, dNUFIS_hexx, chi_hexx, noise_section, type_noise, output_dir, case_name, precond)
    S, dPHI_temp_meas = main_unfold_2D_hexx_solve(PHI_temp, G_matrix, dPHI_temp, keff, group, I_max, J_max, N_hexx, conv_tri, conv_neighbor, TOT, SIGS_reshaped, BC, h, level, D, chi, NUFIS, v, Beff, omega, l, dTOT_hexx, dSIGS_hexx, dNUFIS_hexx, chi_hexx, noise_section, type_noise, map_detector_hexx, output_dir, case_name, precond, tri_indices, x, y)

##### OLD METHODS (INVERSION, ZONING, and SCANNING)
    dPHI_temp_INVERT, dS_unfold_INVERT_temp = main_unfold_2D_hexx_invert(dPHI_temp_meas, dPHI_temp, S, G_matrix, group, I_max, J_max, N_hexx, conv_tri, level, map_detector_hexx, map_zone, output_dir, case_name, tri_indices, x, y, all_triangles)
    dS_unfold_ZONE_temp = main_unfold_2D_hexx_zone(dPHI_temp_meas, dPHI_temp, S, G_matrix, group, I_max, J_max, N_hexx, conv_tri, level, map_detector_hexx, map_zone, output_dir, case_name, tri_indices, x, y, all_triangles)
    dS_unfold_SCAN_temp = main_unfold_2D_hexx_scan(dPHI_temp_meas, dPHI_temp, S, G_matrix, group, I_max, J_max, N_hexx, conv_tri, level, map_detector_hexx, map_zone, output_dir, case_name, tri_indices, x, y, all_triangles)

##### BRUTE FORCE METHOD
    if type_noise == 'FVX' or type_noise == 'FAV':
        print("Brute Force Skipped")
        pass
    else:
        dPHI_temp_BRUTE, dS_unfold_BRUTE_temp = main_unfold_2D_hexx_brute(dPHI_temp_meas, dPHI_temp, S, G_matrix, group, N_hexx, conv_tri, output_dir, case_name, tri_indices, x, y)

#### BACKWARD ELIMINATION METHOD
    dPHI_temp_BACK, dS_unfold_BACK_temp = main_unfold_2D_hexx_back(dPHI_temp_meas, dPHI_temp, S, G_matrix, group, N_hexx, conv_tri, output_dir, case_name, tri_indices, x, y)

#### GREEDY METHOD
#    dPHI_temp_GREEDY, dS_unfold_GREEDY_temp = main_unfold_2D_hexx_greedy(dPHI_temp_meas, dPHI_temp, S, G_matrix, group, N_hexx, conv_tri, output_dir, case_name, tri_indices, x, y)
    dPHI_temp_GREEDY, dS_unfold_GREEDY_temp = main_unfold_2D_hexx_greedy_new(dPHI_temp_meas, dPHI_temp, S, G_matrix, group, N_hexx, conv_tri, output_dir, case_name, tri_indices, x, y)

    ####################################################################################################
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Time elapsed: {elapsed_time:3e} seconds')

if __name__ == "__main__":
    main()