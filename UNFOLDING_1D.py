import numpy as np
import json
import time
import os
import sys

# Prevent .pyc file generation
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True

start_time = time.time()

from SRC.MATRIX_BUILDER import *
from SRC.METHODS import *
from SRC.SOLVERFACTORY import SolverFactory
from SRC.SRC_UNFOLDING_1D import *

#######################################################################################################
# INPUTS
original_sys_path = sys.path.copy()
sys.path.append('../')

#from INPUTS.TASK3_TEST01a_1D1G_1SRC import *
#from INPUTS.TASK3_TEST01b_1D1G_2SRC import *
#from INPUTS.TASK3_TEST01c_1DMG_CSTest03 import *
#from INPUTS.TASK3_TEST01d_1D1G_COMPLEX import *
from INPUTS.TASK3_TEST01e_1D1G_4SRC import *

# Restore the original sys.path
sys.path = original_sys_path

#######################################################################################################
def main():
    start_time = time.time()

    output_dir = f'OUTPUTS/{case_name}/{case_name}_UNFOLDING'

##### Forward Simulation
    solver_type = 'forward'
    os.makedirs(f'{output_dir}/{case_name}_{solver_type.upper()}', exist_ok=True)

    matrix_builder = MatrixBuilderForward1D(group, N, TOT, SIGS, BC, dx, D, chi, NUFIS)
    M, F_FORWARD = matrix_builder.build_forward_matrices()
    solver = SolverFactory.get_solver_power1D(solver_type, group, N, M, F_FORWARD, x, precond, tol=1E-6)

    keff, PHI = solver.solve()
    PHI_reshaped = np.reshape(PHI, (group, N))
    output = {"keff": keff}
    for g in range(len(PHI_reshaped)):
        phi_groupname = f'PHI{g + 1}_{solver_type.upper()}'
        output[phi_groupname] = [val.real for val in PHI_reshaped[g]]

    with open(f'{output_dir}/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_output.json', 'w') as json_file:
        json.dump(output, json_file, indent=4)

##### Adjoint Simulation
    solver_type = 'adjoint'
    os.makedirs(f'{output_dir}/{case_name}_{solver_type.upper()}', exist_ok=True)

    matrix_builder = MatrixBuilderAdjoint1D(group, N, TOT, SIGS, BC, dx, D, chi, NUFIS)
    M, F_ADJOINT = matrix_builder.build_adjoint_matrices()
    solver = SolverFactory.get_solver_power1D(solver_type, group, N, M, F_ADJOINT, x, precond, tol=1E-6)

    keff, PHI_ADJ = solver.solve()
    PHI_ADJ_reshaped = np.reshape(PHI_ADJ, (group, N))
    output = {"keff": keff}
    for g in range(len(PHI_ADJ_reshaped)):
        phi_groupname = f'PHI{g + 1}_{solver_type.upper()}'
        output[phi_groupname] = [val.real for val in PHI_reshaped[g]]

    with open(f'{output_dir}/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_output.json', 'w') as json_file:
        json.dump(output, json_file, indent=4)

##### BASE (Noise, Green's function, and solve)
    dPHI = main_unfold_1D_noise(PHI, keff, group, N, TOT, SIGS, BC, dx, D, chi, NUFIS, precond, v, Beff, omega, l, dTOT, dSIGS, dNUFIS, dSOURCE, output_dir, case_name, x)
    G_matrix = main_unfold_1D_green(PHI, keff, group, N, TOT, SIGS, BC, dx, D, chi, NUFIS, precond, v, Beff, omega, l, dTOT, dSIGS, dNUFIS, dSOURCE, output_dir, case_name, x)
    S, dPHI_meas = main_unfold_1D_solve(PHI, G_matrix, dPHI, keff, group, N, TOT, SIGS, BC, dx, D, chi, NUFIS, precond, v, Beff, omega, l, dTOT, dSIGS, dNUFIS, dSOURCE, map_detector, output_dir, case_name, x)

###### OLD METHODS (INVERSION, and SCANNING)
#    dPHI_INVERT, dS_unfold_INVERT = main_unfold_1D_invert(dPHI_meas, dPHI, S, G_matrix, group, N, map_detector, output_dir, case_name, x)
#    dS_unfold_SCAN = main_unfold_1D_scan(dPHI_meas, dPHI, S, G_matrix, group, N, map_detector, output_dir, case_name, x)
#
###### BRUTE FORCE METHOD
#    dPHI_BRUTE, dS_unfold_BRUTE = main_unfold_1D_brute(dPHI_meas, dPHI, S, G_matrix, group, N, output_dir, case_name, x)
#
##### GREEDY METHOD
    dPHI_GREEDY, dS_unfold_GREEDY = main_unfold_1D_greedy_new(dPHI_meas, dPHI, S, G_matrix, group, N, output_dir, case_name, x)

##### GREEDY_OPTIMIZED METHOD
    dPHI_GREEDY, dS_unfold_GREEDY = main_unfold_1D_greedy_optimized(dPHI_meas, dPHI, S, G_matrix, group, N, output_dir, case_name, x)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Time elapsed: {elapsed_time:3e} seconds')

if __name__ == "__main__":
    main()