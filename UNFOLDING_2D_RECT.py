import numpy as np
import json
import time
import os
import sys
from scipy.integrate import trapezoid
import scipy.linalg
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
from SRC.XSPROCESS_2D_RECT import *
from SRC.SRC_UNFOLDING_2D_RECT import *

#######################################################################################################
# INPUTS
original_sys_path = sys.path.copy()
sys.path.append('../')

#from INPUTS.OBJECTIVES45_TEST01_2DMG_BIBLIS_AVS import *
#from INPUTS.OBJECTIVES45_TEST02_2DMG_BIBLIS3_FAV import *
#from INPUTS.OBJECTIVES45_TEST09_2DMG_BIBLIS_AVS3S import *

#from INPUTS.TASK3_TEST02a_2DMG_PWRMOX1_1SRC_AVS_NONCENTER import *
from INPUTS.TASK3_TEST02b_2DMG_PWRMOX1_2SRC_AVS_NONCENTER import *
#from INPUTS.TASK3_TEST02c_2DMG_PWRMOX1_3SRC_AVS_NONCENTER import *
#from INPUTS.TASK3_TEST02d_2DMG_PWRMOX1_4SRC_AVS_NONCENTER import *
#from INPUTS.TASK3_TEST02e_2DMG_PWRMOX1_5SRC_AVS_NONCENTER import *
#from INPUTS.TASK3_TEST02f_2DMG_PWRMOX1_6SRC_AVS_NONCENTER import *

# Restore the original sys.path
sys.path = original_sys_path

#######################################################################################################
def main():
    start_time = time.time()

    output_dir = f'OUTPUTS/{case_name}/{case_name}_UNFOLDING'

##### Forward Simulation
    solver_type = 'forward'
    os.makedirs(f'{output_dir}/{case_name}_{solver_type.upper()}', exist_ok=True)
    conv = convert_index_2D_rect(D, I_max, J_max)
    conv_array = np.array(conv)

    matrix_builder = MatrixBuilderForward2DRect(group, N, conv, TOT, SIGS_reshaped, BC, dx, dy, D, chi, NUFIS)
    M, F_FORWARD = matrix_builder.build_forward_matrices()

    solver = SolverFactory.get_solver_power2DRect(solver_type, group, N, conv, M, F_FORWARD, dx, dy, precond, tol=1E-10)
    keff, PHI_temp = solver.solve()
    PHI, PHI_reshaped, PHI_reshaped_plot = PostProcessor.postprocess_power2DRect(PHI_temp, conv, group, N, I_max, J_max)

    output = {"keff": keff.real}
    for g in range(len(PHI_reshaped)):
        phi_groupname = f'PHI{g + 1}_{solver_type.upper()}'
        output[phi_groupname] = [val.real for val in PHI_reshaped[g]]

    with open(f'{output_dir}/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_output.json', 'w') as json_file:
        json.dump(output, json_file, indent=4)

##### Adjoint Simulation
    solver_type = 'adjoint'
    os.makedirs(f'{output_dir}/{case_name}_{solver_type.upper()}', exist_ok=True)
    conv = convert_index_2D_rect(D, I_max, J_max)
    conv_array = np.array(conv)

    matrix_builder = MatrixBuilderAdjoint2DRect(group, N, conv, TOT, SIGS_reshaped, BC, dx, dy, D, chi, NUFIS)
    M, F_ADJOINT = matrix_builder.build_adjoint_matrices()

    solver = SolverFactory.get_solver_power2DRect(solver_type, group, N, conv, M, F_ADJOINT, dx, dy, precond, tol=1E-10)
    keff, PHI_ADJ_temp = solver.solve()
    PHI_ADJ, PHI_ADJ_reshaped, PHI_ADJ_reshaped_plot = PostProcessor.postprocess_power2DRect(PHI_ADJ_temp, conv, group, N, I_max, J_max)

    output = {"keff": keff.real}
    for g in range(len(PHI_ADJ_reshaped)):
        phi_groupname = f'PHI{g + 1}_{solver_type.upper()}'
        output[phi_groupname] = [val.real for val in PHI_ADJ_reshaped[g]]

    with open(f'{output_dir}/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_output.json', 'w') as json_file:
        json.dump(output, json_file, indent=4)

##### BASE (Noise, Green's function, and solve)
    dPHI_temp = main_unfold_2D_rect_noise(PHI_temp, keff, group, N, I_max, J_max, TOT, SIGS_reshaped, BC, dx, dy, D, chi, NUFIS, precond, v, Beff, omega, l, dTOT, dSIGS_reshaped, dNUFIS, map_detector, output_dir, case_name, x, y)
    G_matrix = main_unfold_2D_rect_green(PHI_temp, keff, group, N, I_max, J_max, TOT, SIGS_reshaped, BC, dx, dy, D, chi, NUFIS, precond, v, Beff, omega, l, dTOT, dSIGS_reshaped, dNUFIS, map_detector, output_dir, case_name, x, y)
    S, dPHI_temp_meas = main_unfold_2D_rect_solve(PHI_temp, G_matrix, dPHI_temp, keff, group, N, I_max, J_max, TOT, SIGS_reshaped, BC, dx, dy, D, chi, NUFIS, precond, v, Beff, omega, l, dTOT, dSIGS_reshaped, dNUFIS, map_detector, output_dir, case_name, x, y)

##### OLD METHODS (INVERSION, ZONING, and SCANNING)
#    dPHI_temp_INVERT, dS_unfold_INVERT_temp = main_unfold_2D_rect_invert(dPHI_temp_meas, dPHI_temp, S, G_matrix, group, N, I_max, J_max, conv, map_detector, map_zone, output_dir, case_name, x, y)
#    dS_unfold_ZONE_temp = main_unfold_2D_rect_zone(dPHI_temp_meas, dPHI_temp, S, G_matrix, group, N, I_max, J_max, conv, map_detector, map_zone, output_dir, case_name, x, y)
#    dS_unfold_SCAN_temp = main_unfold_2D_rect_scan(dPHI_temp_meas, dPHI_temp, S, G_matrix, group, N, I_max, J_max, conv, map_detector, map_zone, output_dir, case_name, x, y)
#
##### BRUTE FORCE METHOD
#    if type_noise == 'FVX' or type_noise == 'FAV':
#        print("Brute Force Skipped")
#        pass
#    else:
#        dPHI_temp_BRUTE, dS_unfold_BRUTE_temp = main_unfold_2D_rect_brute(dPHI_temp_meas, dPHI_temp, S, G_matrix, group, N, I_max, J_max, conv, map_detector, output_dir, case_name, x, y)

#### GREEDY METHOD
    dPHI_temp_GREEDY, dS_unfold_GREEDY_temp = main_unfold_2D_rect_greedy_optimized(dPHI_temp_meas, dPHI_temp, S, G_matrix, group, N, I_max, J_max, conv, output_dir, case_name, x, y)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Time elapsed: {elapsed_time:3e} seconds')

if __name__ == "__main__":
    main()