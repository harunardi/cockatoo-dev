import numpy as np
import json
import time
import os
import sys
from scipy.integrate import trapezoid

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
# INPUTS
original_sys_path = sys.path.copy()
sys.path.append('../')

from INPUTS.OBJECTIVES3_TEST01_2DMG_BIBLIS_AVS import *
#from INPUTS.OBJECTIVES3_TEST01_2DMG_BIBLIS_CENTER_AVS import *
#from INPUTS.OBJECTIVES3_TEST02_2DMG_BIBLIS_FAV import *
#from INPUTS.OBJECTIVES3_TEST02_2DMG_BIBLIS_CENTER_FAV import *
#from INPUTS.OBJECTIVES3_TEST03_2DTriMG_HTTR2G_AVS import *
#from INPUTS.OBJECTIVES3_TEST04_2DTriMG_HTTR2G_FAV import *
#from INPUTS.OBJECTIVES3_TEST05_3DMG_CSTest09_AVS import *
#from INPUTS.OBJECTIVES3_TEST05_3DMG_CSTest09_CENTER_AVS import *
#from INPUTS.OBJECTIVES3_TEST06_3DMG_CSTest09_FAV import *
#from INPUTS.OBJECTIVES3_TEST06_3DMG_CSTest09_CENTER_FAV import *
#from INPUTS.OBJECTIVES3_TEST07_3DTriMG_HTTR_AVS import *
#from INPUTS.OBJECTIVES3_TEST08_3DTriMG_HTTR_FAV import *
#from INPUTS.OBJECTIVES3_TEST09_3DMG_LangenOBJECTIbuch_AVS import *

# Restore the original sys.path
sys.path = original_sys_path

#######################################################################################################
def main():
    start_time = time.time()

    if geom_type =='1D':
        output_dir = f'OUTPUTS/{case_name}'
        x = globals().get("x")
        dx = globals().get("dx")
        N = globals().get("N")
        group = globals().get("group")
        D = globals().get("D")
        TOT = globals().get("TOT")
        SIGS = globals().get("SIGS")
        chi = globals().get("chi")
        NUFIS = globals().get("NUFIS")
        BC = globals().get("BC")

        # Forward Simulation
        solver_type = 'forward'
        os.makedirs(f'{output_dir}/{case_name}_PK_COMPONENTS/{case_name}_{solver_type.upper()}', exist_ok=True)
        matrix_builder = MatrixBuilderForward1D(group, N, TOT, SIGS, BC, dx, D, chi, NUFIS)
        M, F_FORWARD = matrix_builder.build_forward_matrices()
        solver = SolverFactory.get_solver_power1D(solver_type, group, N, M, F_FORWARD, dx, precond, tol=1E-6)
        keff, PHI = solver.solve()
        PHI_reshaped = np.reshape(PHI, (group, N))

        output = {"keff": keff}
        for g in range(len(PHI_reshaped)):
            phi_groupname = f'PHI{g + 1}_{solver_type.upper()}'
            output[phi_groupname] = [val.real for val in PHI_reshaped[g]]

        with open(f'{output_dir}/{case_name}_PK_COMPONENTS/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_output.json', 'w') as json_file:
            json.dump(output, json_file, indent=4)

        # Adjoint Simulation
        solver_type = 'adjoint'
        os.makedirs(f'{output_dir}/{case_name}_PK_COMPONENTS/{case_name}_{solver_type.upper()}', exist_ok=True)
        matrix_builder = MatrixBuilderAdjoint1D(group, N, TOT, SIGS, BC, dx, D, chi, NUFIS)
        M, F_ADJOINT = matrix_builder.build_adjoint_matrices()
        solver = SolverFactory.get_solver_power1D(solver_type, group, N, M, F_ADJOINT, dx, precond, tol=1E-6)
        keff, PHI_ADJ = solver.solve()
        PHI_ADJ_reshaped = np.reshape(PHI_ADJ, (group, N))

        output = {"keff": keff}
        for g in range(len(PHI_ADJ_reshaped)):
            phi_groupname = f'PHI{g + 1}_{solver_type.upper()}'
            output[phi_groupname] = [val.real for val in PHI_reshaped[g]]

        with open(f'{output_dir}/{case_name}_PK_COMPONENTS/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_output.json', 'w') as json_file:
            json.dump(output, json_file, indent=4)

        # Noise Simulation
        solver_type = 'noise'
        v = globals().get("v")
        Beff = globals().get("Beff")
        omega = globals().get("omega")
        l = globals().get("l")
        dTOT = globals().get("dTOT")
        dSIGS = globals().get("dSIGS")
        dNUFIS = globals().get("dNUFIS")
        dSOURCE = globals().get("dSOURCE")
        os.makedirs(f'{output_dir}/{case_name}_PK_COMPONENT/{case_name}_{solver_type.upper()}', exist_ok=True)
        with open(f'{output_dir}/{case_name}_PK_COMPONENT/{case_name}_FORWARD/{case_name}_FORWARD_output.json', 'r') as json_file:
            forward_output = json.load(json_file)
        keff = forward_output["keff"]
        PHI = []
        for i in range(group):
            phi_key = f"PHI{i+1}_FORWARD"
            PHI.extend(forward_output[phi_key])

        matrix_builder = MatrixBuilderNoise1D(group, N, TOT, SIGS, BC, dx, D, chi, NUFIS, keff, v, Beff, omega, l, dTOT, dSIGS, dNUFIS)
        M, dS = matrix_builder.build_noise_matrices()

        solver = SolverFactory.get_solver_fixed1D(solver_type, group, N, M, dS, dSOURCE, PHI, dx, precond, tol=1e-06)

        dPHI = solver.solve()
        dPHI_reshaped = np.reshape(dPHI, (group, N))
        output = {}
        for g in range(len(dPHI_reshaped)):
            dPHI_groupname = f'dPHI{g + 1}'
            dPHI_list = [{"real": x.real, "imaginary": x.imag} for x in dPHI_reshaped[g]]
            output[dPHI_groupname] = dPHI_list
    
        with open(f'{output_dir}/{case_name}_PK_COMPONENTS/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_output.json', 'w') as json_file:
            json.dump(output, json_file, indent=4)

        NUMER = 0
        DENOM = 0
        for g in range(group):
            NUMER += trapezoid((1/v[g]) * (PHI_ADJ_reshaped[g] * dPHI_reshaped[g]), dx=dx, axis = 0)
            DENOM += trapezoid((1/v[g]) * (PHI_ADJ_reshaped[g] * PHI_reshaped[g]), dx=dx, axis = 0)

        dPOWER = (NUMER / DENOM)

        S_NUMER = dS.dot(PHI)
        S_NUMER_reshaped = np.reshape(S_NUMER, (group, N))
        S_DENOM = F_FORWARD.dot(PHI)
        S_DENOM_reshaped = np.reshape(S_DENOM, (group, N))

        NUMER = 0
        DENOM = 0
        for g in range(group):
            NUMER += trapezoid(PHI_ADJ_reshaped[g] * S_NUMER_reshaped[g], dx=dx, axis = 0)
            DENOM += trapezoid(PHI_ADJ_reshaped[g] * S_DENOM_reshaped[g], dx=dx, axis = 0)

        dRHO = (NUMER / DENOM)

        dPHI_pk = dPOWER * PHI
        dPHI_spatial = dPHI - dPHI_pk

        ABS_dPHI_pk = np.abs(dPHI_pk)
        ABS_dPHI_spatial = np.abs(dPHI_spatial)

        ABS_dPHI_pk_plot = ABS_dPHI_pk.reshape(group, N)
        ABS_dPHI_spatial_plot = ABS_dPHI_spatial.reshape(group, N)

        for g in range(group):
            Utils.plot_1D_fixed(solver_type, dPHI_reshaped[g], x, g, output_dir=output_dir, varname=f'dPHI', case_name=case_name, title=f'1D Plot of dPHI{g+1}')
            Utils.plot_1D_fixed(solver_type, ABS_dPHI_pk_plot[g], x, g, output_dir=output_dir, varname=f'dPHI_pk', case_name=case_name, title=f'1D Plot of dPHI{g+1}_pk')
            Utils.plot_1D_fixed(solver_type, ABS_dPHI_spatial_plot[g], x, g, output_dir=output_dir, varname=f'dPHI_spatial', case_name=case_name, title=f'1D Plot of dPHI{g+1}_spatial')

        time_step = np.linspace(0,10, 1001)
        dRHO_time = np.abs(dRHO) * np.cos(2 * np.pi * f * time_step + np.angle(dRHO)) * 1E+5  # in pcm

        plt.figure(figsize=(8, 4))
        plt.plot(time_step, dRHO_time)
        plt.title("dRHO vs. Time")
        plt.xlabel("Time (s)")
        plt.ylabel(r"$\delta \rho$ (pcm)")
        plt.grid(True)
        plt.savefig(f'{output_dir}/{case_name}_PK_COMPONENTS/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_dRHO_time.png')

    elif geom_type =='2D rectangular':
        output_dir = f'OUTPUTS/{case_name}'
        x = globals().get("x")
        y = globals().get("y")
        dx = globals().get("dx")
        dy = globals().get("dy")
        I_max = globals().get("I_max")
        J_max = globals().get("J_max")
        N = globals().get("N")
        group = globals().get("group")
        D = globals().get("D")
        TOT = globals().get("TOT")
        SIGS_reshaped = globals().get("SIGS_reshaped")
        chi = globals().get("chi")
        NUFIS = globals().get("NUFIS")
        BC = globals().get("BC")

        # Forward Simulation
        solver_type = 'forward'
        os.makedirs(f'{output_dir}/{case_name}_PK_COMPONENTS/{case_name}_{solver_type.upper()}', exist_ok=True)
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

        with open(f'{output_dir}/{case_name}_PK_COMPONENTS/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_output.json', 'w') as json_file:
            json.dump(output, json_file, indent=4)

        PHI_temp_reshaped = np.reshape(PHI_temp, (group, max(conv)))

        # Adjoint Simulation
        solver_type = 'adjoint'
        os.makedirs(f'{output_dir}/{case_name}_PK_COMPONENTS/{case_name}_{solver_type.upper()}', exist_ok=True)
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

        with open(f'{output_dir}/{case_name}_PK_COMPONENTS/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_output.json', 'w') as json_file:
            json.dump(output, json_file, indent=4)

        PHI_ADJ_temp_reshaped = np.reshape(PHI_ADJ_temp, (group, max(conv)))

        # Noise Simulation
        solver_type = 'noise'
        v = globals().get("v")
        Beff = globals().get("Beff")
        omega = globals().get("omega")
        l = globals().get("l")
        dTOT = globals().get("dTOT")
        dSIGS_reshaped = globals().get("dSIGS_reshaped")
        dNUFIS = globals().get("dNUFIS")
        os.makedirs(f'{output_dir}/{case_name}_PK_COMPONENTS/{case_name}_{solver_type.upper()}', exist_ok=True)
        conv = convert_index_2D_rect(D, I_max, J_max)
        conv_array = np.array(conv)

        with open(f'{output_dir}/{case_name}_PK_COMPONENTS/{case_name}_FORWARD/{case_name}_FORWARD_output.json', 'r') as json_file:
            forward_output = json.load(json_file)
        keff = forward_output["keff"]
        PHI_all = []
        for i in range(group):
            phi_key = f"PHI{i+1}_FORWARD"
            PHI_all.append(forward_output[phi_key])

        PHI = np.zeros(max(conv) * group)
        for g in range(group):
            PHI_indices = g * max(conv) + (conv_array - 1)
            PHI[PHI_indices] = PHI_all[g]

        matrix_builder = MatrixBuilderNoise2DRect(group, N, conv, TOT, SIGS_reshaped, BC, dx, dy, D, chi, NUFIS, keff, v, Beff, omega, l, dTOT, dSIGS_reshaped, dNUFIS)
        M, dS = matrix_builder.build_noise_matrices()

        solver = SolverFactory.get_solver_fixed2DRect(solver_type, group, N, conv, M, dS, PHI, dx, dy, precond, tol=1e-10)

        dPHI_temp = solver.solve()
        dPHI, dPHI_reshaped, dPHI_reshaped_plot = PostProcessor.postprocess_fixed2DRect(dPHI_temp, conv, group, N, I_max, J_max)
        output = {}
        for g in range(len(dPHI_reshaped)):
            dPHI_groupname = f'dPHI{g + 1}'
            dPHI_list = [{"real": x.real, "imaginary": x.imag} for x in dPHI_reshaped[g]]
            output[dPHI_groupname] = dPHI_list
    
        with open(f'{output_dir}/{case_name}_PK_COMPONENTS/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_output.json', 'w') as json_file:
            json.dump(output, json_file, indent=4)

        dPHI_temp_reshaped = np.reshape(dPHI_temp, (group, max(conv)))

        v1_PP = v1[0][0]
        v2_PP = v2[0][0]
        v_new = [v1_PP, v2_PP]

        NUMER = 0
        DENOM = 0
        for g in range(group):
            NUMER += trapezoid((1/v_new[g]) * (PHI_ADJ_temp_reshaped[g] * dPHI_temp_reshaped[g]), dx=dx*dy, axis = 0)
            DENOM += trapezoid((1/v_new[g]) * (PHI_ADJ_temp_reshaped[g] * PHI_temp_reshaped[g]), dx=dx*dy, axis = 0)

        dPOWER = (NUMER / DENOM)

        S_NUMER = dS.dot(PHI)
        S_NUMER_reshaped = np.reshape(S_NUMER, (group, max(conv)))
        S_DENOM = F_FORWARD.dot(PHI)
        S_DENOM_reshaped = np.reshape(S_DENOM, (group, max(conv)))
        S_NUMER_reshaped = np.nan_to_num(S_NUMER_reshaped, nan=0)
        S_DENOM_reshaped = np.nan_to_num(S_DENOM_reshaped, nan=0)

        NUMER = 0
        DENOM = 0
        for g in range(group):
            NUMER += trapezoid(PHI_ADJ_temp_reshaped[g] * S_NUMER_reshaped[g], dx=dx*dy, axis = 0)
            DENOM += trapezoid(PHI_ADJ_temp_reshaped[g] * S_DENOM_reshaped[g], dx=dx*dy, axis = 0)

        dRHO = (NUMER / DENOM)

        dPHI_pk_temp = dPOWER * PHI
        dPHI_spatial_temp = dPHI_temp - dPHI_pk_temp

        dPHI_pk, dPHI_pk_reshaped, dPHI_pk_reshaped_plot = PostProcessor.postprocess_fixed2DRect(dPHI_pk_temp, conv, group, N, I_max, J_max)
        output = {}
        for g in range(len(dPHI_pk_reshaped)):
            dPHI_pk_groupname = f'dPHI{g + 1}_pk'
            dPHI_pk_list = [{"real": x.real, "imaginary": x.imag} for x in dPHI_pk_reshaped[g]]
            output[dPHI_pk_groupname] = dPHI_pk_list
        with open(f'{output_dir}/{case_name}_PK_COMPONENTS/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_pk_output.json', 'w') as json_file:
            json.dump(output, json_file, indent=4)

        dPHI_spatial, dPHI_spatial_reshaped, dPHI_spatial_reshaped_plot = PostProcessor.postprocess_fixed2DRect(dPHI_spatial_temp, conv, group, N, I_max, J_max)
        output = {}
        for g in range(len(dPHI_spatial_reshaped)):
            dPHI_spatial_groupname = f'dPHI{g + 1}_spatial'
            dPHI_spatial_list = [{"real": x.real, "imaginary": x.imag} for x in dPHI_spatial_reshaped[g]]
            output[dPHI_spatial_groupname] = dPHI_spatial_list
        with open(f'{output_dir}/{case_name}_PK_COMPONENTS/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_spatial_output.json', 'w') as json_file:
            json.dump(output, json_file, indent=4)

        for g in range(group):
            Utils.plot_2D_rect_fixed(solver_type, dPHI_reshaped_plot[g], x, y, g+1, cmap='viridis', output_dir=f'{output_dir}/{case_name}_PK_COMPONENTS', varname=f'dPHI', case_name=case_name, title=f'1D Plot of dPHI{g+1}', process_data='magnitude')
            Utils.plot_2D_rect_fixed(solver_type, dPHI_reshaped_plot[g], x, y, g+1, cmap='viridis', output_dir=f'{output_dir}/{case_name}_PK_COMPONENTS', varname=f'dPHI', case_name=case_name, title=f'1D Plot of dPHI{g+1}', process_data='phase')
            Utils.plot_2D_rect_fixed(solver_type, dPHI_pk_reshaped_plot[g], x, y, g+1, cmap='viridis', output_dir=f'{output_dir}/{case_name}_PK_COMPONENTS', varname=f'dPHI_pk', case_name=case_name, title=f'1D Plot of dPHI{g+1}_pk', process_data='magnitude')
            Utils.plot_2D_rect_fixed(solver_type, dPHI_spatial_reshaped_plot[g], x, y, g+1, cmap='viridis', output_dir=f'{output_dir}/{case_name}_PK_COMPONENTS', varname=f'dPHI_spatial', case_name=case_name, title=f'1D Plot of dPHI{g+1}_spatial', process_data='magnitude')
            Utils.plot_2D_rect_fixed(solver_type, dPHI_pk_reshaped_plot[g], x, y, g+1, cmap='viridis', output_dir=f'{output_dir}/{case_name}_PK_COMPONENTS', varname=f'dPHI_pk', case_name=case_name, title=f'1D Plot of dPHI{g+1}_pk', process_data='phase')
            Utils.plot_2D_rect_fixed(solver_type, dPHI_spatial_reshaped_plot[g], x, y, g+1, cmap='viridis', output_dir=f'{output_dir}/{case_name}_PK_COMPONENTS', varname=f'dPHI_spatial', case_name=case_name, title=f'1D Plot of dPHI{g+1}_spatial', process_data='phase')

        time_step = np.linspace(0,10, 1001)
        dRHO_time = np.abs(dRHO) * np.cos(2 * np.pi * f * time_step + np.angle(dRHO)) * 1E+5  # in pcm

        plt.figure(figsize=(8, 4))
        plt.plot(time_step, dRHO_time)
        plt.title("dRHO vs. Time")
        plt.xlabel("Time (s)")
        plt.ylabel(r"$\delta \rho$ (pcm)")
        plt.grid(True)
        plt.savefig(f'{output_dir}/{case_name}_PK_COMPONENTS/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_dRHO_time.png')

    elif geom_type =='2D triangular':
        h = globals().get("h")
        s = globals().get("s")
        N_hexx = globals().get("N_hexx")
        level = globals().get("level")
        I_max = globals().get("I_max")
        J_max = globals().get("J_max")
        N = globals().get("N")
        group = globals().get("group")
        D = globals().get("D")
        TOT = globals().get("TOT")
        SIGS_reshaped = globals().get("SIGS_reshaped")
        chi = globals().get("chi")
        NUFIS = globals().get("NUFIS")
        BC = globals().get("BC")
        input_name = globals().get("input_name")
        output_dir = f'OUTPUTS/{input_name}'

        # Forward Simulation
        solver_type = 'forward'
        os.makedirs(f'{output_dir}/{case_name}_PK_COMPONENTS/{case_name}_{solver_type.upper()}', exist_ok=True)
        conv_hexx = convert_2D_hexx(I_max, J_max, D)
        conv_tri, conv_hexx_ext = convert_2D_tri(I_max, J_max, conv_hexx, level)
        conv_tri_array = np.array(conv_tri)
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

        with open(f'{output_dir}/{case_name}_PK_COMPONENTS/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_output.json', 'w') as json_file:
            json.dump(output, json_file, indent=4)

        # Adjoint Simulation
        solver_type = 'adjoint'
        os.makedirs(f'{output_dir}/{case_name}_PK_COMPONENTS/{case_name}_{solver_type.upper()}', exist_ok=True)

        matrix_builder = MatrixBuilderAdjoint2DHexx(group, I_max, J_max, conv_tri, conv_neighbor, TOT, SIGS_reshaped, BC, h, level, D, chi, NUFIS)
        M, F_ADJOINT = matrix_builder.build_adjoint_matrices()

        solver = SolverFactory.get_solver_power2DHexx(solver_type, group, conv_tri, M, F_ADJOINT, h, precond, tol=1E-10)
        keff, PHI_ADJ_temp = solver.solve()
        PHI_ADJ, PHI_ADJ_reshaped, PHI_ADJ_temp_reshaped = PostProcessor.postprocess_power2DHexx(PHI_ADJ_temp, conv_tri, group, N_hexx)

        output = {"keff": keff.real}
        for g in range(len(PHI_ADJ_reshaped)):
            phi_groupname = f'PHI{g + 1}_{solver_type.upper()}'
            output[phi_groupname] = [val.real for val in PHI_ADJ_reshaped[g]]

        with open(f'{output_dir}/{case_name}_PK_COMPONENTS/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_output.json', 'w') as json_file:
            json.dump(output, json_file, indent=4)

        # Noise Simulation
        solver_type = 'noise'
        v = globals().get("v")
        Beff = globals().get("Beff")
        omega = globals().get("omega")
        l = globals().get("l")
        dTOT = globals().get("dTOT")
        dSIGS_reshaped = globals().get("dSIGS_reshaped")
        dNUFIS = globals().get("dNUFIS")
        noise_section = globals().get("noise_section")
        os.makedirs(f'{output_dir}/{case_name}_PK_COMPONENTS/{case_name}_{solver_type.upper()}', exist_ok=True)

        with open(f'{output_dir}/{case_name}_PK_COMPONENTS/{case_name}_FORWARD/{case_name}_FORWARD_output.json', 'r') as json_file:
            forward_output = json.load(json_file)
        keff = forward_output["keff"]
        PHI_all = []
        for i in range(group):
            phi_key = f"PHI{i+1}_FORWARD"
            PHI_all.append(forward_output[phi_key])

        PHI = np.zeros(max(conv_tri) * group)
        for g in range(group):
            for n in range(N_hexx):
                if conv_tri[n] != 0:
                    PHI[g * max(conv_tri) + conv_tri[n]-1] = PHI_all[g][n]

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
                print('Vibrating Assembly type noise only works if level => 2. Changing level to 4')

        hex_centers, hex_vertices = generate_pointy_hex_grid(s, I_max, J_max)
        triangle_neighbors_global = find_triangle_neighbors_2D(all_triangles, precision=6)

        if type_noise == 'FXV':
            dTOT_hexx, dNUFIS_hexx = XS2D_FXV(level, group, s, J_max, I_max, dTOT, dNUFIS, fav_strength, diff_X_ABS, diff_X_NUFIS)
        elif type_noise == 'FAV':
            dTOT_hexx, dNUFIS_hexx = XS2D_FAV(level, group, s, J_max, I_max, dTOT, dNUFIS, fav_strength, diff_X_ABS, diff_X_NUFIS)

        matrix_builder = MatrixBuilderNoise2DHexx(group, I_max, J_max, N_hexx, conv_tri, conv_neighbor, TOT, SIGS_reshaped, BC, h, level, D, chi, NUFIS, keff, v, Beff, omega, l, dTOT_hexx, dSIGS_hexx, chi_hexx, dNUFIS_hexx, noise_section, type_noise)
        M, dS = matrix_builder.build_noise_matrices()
        solver = SolverFactory.get_solver_fixed2DHexx(solver_type, group, conv_tri, M, dS, PHI, precond, tol=1e-10)

        dPHI_temp = solver.solve()
        dPHI, dPHI_reshaped, dPHI_temp_reshaped = PostProcessor.postprocess_fixed2DHexx(dPHI_temp, conv_tri, group, N_hexx)

        output = {}
        for g in range(len(dPHI_reshaped)):
            dPHI_groupname = f'dPHI{g + 1}'
            dPHI_list = [{"real": x.real, "imaginary": x.imag} for x in dPHI_reshaped[g]]
            output[dPHI_groupname] = dPHI_list

        with open(f'{output_dir}/{case_name}_PK_COMPONENTS/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_output.json', 'w') as json_file:
            json.dump(output, json_file, indent=4)

        v1_PP = v1
        v2_PP = v2
        v_new = [v1_PP, v2_PP]

        NUMER = 0
        DENOM = 0
        for g in range(group):
            NUMER += trapezoid((1/v_new[g]) * (PHI_ADJ_temp_reshaped[g] * dPHI_temp_reshaped[g]), dx=h**2/4*np.sqrt(3), axis = 0)
            DENOM += trapezoid((1/v_new[g]) * (PHI_ADJ_temp_reshaped[g] * PHI_temp_reshaped[g]), dx=h**2/4*np.sqrt(3), axis = 0)
        dPOWER = (NUMER / DENOM)

        S_NUMER = dS.dot(PHI)
        S_NUMER_reshaped = np.reshape(S_NUMER, (group, max(conv_tri)))
        S_DENOM = F_FORWARD.dot(PHI)
        S_DENOM_reshaped = np.reshape(S_DENOM, (group, max(conv_tri)))
        S_NUMER_reshaped = np.nan_to_num(S_NUMER_reshaped, nan=0)
        S_DENOM_reshaped = np.nan_to_num(S_DENOM_reshaped, nan=0)

        NUMER = 0
        DENOM = 0
        for g in range(group):
            NUMER += trapezoid(PHI_ADJ_temp_reshaped[g] * S_NUMER_reshaped[g], dx=h**2/4*np.sqrt(3), axis = 0)
            DENOM += trapezoid(PHI_ADJ_temp_reshaped[g] * S_DENOM_reshaped[g], dx=h**2/4*np.sqrt(3), axis = 0)
        dRHO = (NUMER / DENOM)

        dPHI_pk_temp = dPOWER * PHI
        dPHI_spatial_temp = dPHI_temp - dPHI_pk_temp

        dPHI_pk, dPHI_pk_reshaped, dPHI_pk_temp_reshaped = PostProcessor.postprocess_fixed2DHexx(dPHI_pk_temp, conv_tri, group, N_hexx)
        output = {}
        for g in range(len(dPHI_pk_reshaped)):
            dPHI_pk_groupname = f'dPHI{g + 1}_pk'
            dPHI_pk_list = [{"real": x.real, "imaginary": x.imag} for x in dPHI_pk_reshaped[g]]
            output[dPHI_pk_groupname] = dPHI_pk_list

        with open(f'{output_dir}/{case_name}_PK_COMPONENTS/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_pk_output.json', 'w') as json_file:
            json.dump(output, json_file, indent=4)

        dPHI_spatial, dPHI_spatial_reshaped, dPHI_spatial_temp_reshaped = PostProcessor.postprocess_fixed2DHexx(dPHI_spatial_temp, conv_tri, group, N_hexx)
        output = {}
        for g in range(len(dPHI_spatial_reshaped)):
            dPHI_spatial_groupname = f'dPHI{g + 1}_spatial'
            dPHI_spatial_list = [{"real": x.real, "imaginary": x.imag} for x in dPHI_spatial_reshaped[g]]
            output[dPHI_spatial_groupname] = dPHI_spatial_list

        with open(f'{output_dir}/{case_name}_PK_COMPONENTS/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_spatial_output.json', 'w') as json_file:
            json.dump(output, json_file, indent=4)

        dPHI_pk_plot = dPHI_pk_temp.reshape(group, max(conv_tri))
        dPHI_spatial_plot = dPHI_spatial_temp.reshape(group, max(conv_tri))

        for g in range(group):
            plot_triangular(dPHI_temp_reshaped[g], x, y, tri_indices, g+1, cmap='viridis', varname='dPHI', title=f'2D Plot of dPHI{g+1} Hexx Magnitude', case_name=case_name, output_dir=f'{output_dir}/{case_name}_PK_COMPONENTS', solve=solver_type.upper(), process_data="magnitude")
            plot_triangular(dPHI_temp_reshaped[g], x, y, tri_indices, g+1, cmap='viridis', varname='dPHI', title=f'2D Plot of dPHI{g+1} Hexx Phase', case_name=case_name, output_dir=f'{output_dir}/{case_name}_PK_COMPONENTS', solve=solver_type.upper(), process_data="phase")
            plot_triangular(dPHI_pk_plot[g], x, y, tri_indices, g+1, cmap='viridis', varname='dPHI_pk', title=f'2D Plot of dPHI{g+1}_pk Hexx Magnitude', case_name=case_name, output_dir=f'{output_dir}/{case_name}_PK_COMPONENTS', solve=solver_type.upper(), process_data="magnitude")
            plot_triangular(dPHI_pk_plot[g], x, y, tri_indices, g+1, cmap='viridis', varname='dPHI_pk', title=f'2D Plot of dPHI{g+1}_pk Hexx Phase', case_name=case_name, output_dir=f'{output_dir}/{case_name}_PK_COMPONENTS', solve=solver_type.upper(), process_data="phase")
            plot_triangular(dPHI_spatial_plot[g], x, y, tri_indices, g+1, cmap='viridis', varname='dPHI_spatial', title=f'2D Plot of dPHI{g+1}_spatial Hexx Magnitude', case_name=case_name, output_dir=f'{output_dir}/{case_name}_PK_COMPONENTS', solve=solver_type.upper(), process_data="magnitude")
            plot_triangular(dPHI_spatial_plot[g], x, y, tri_indices, g+1, cmap='viridis', varname='dPHI_spatial', title=f'2D Plot of dPHI{g+1}_spatial Hexx Phase', case_name=case_name, output_dir=f'{output_dir}/{case_name}_PK_COMPONENTS', solve=solver_type.upper(), process_data="phase")

        time_step = np.linspace(0,10, 1001)
        dRHO_time = np.abs(dRHO) * np.cos(2 * np.pi * f * time_step + np.angle(dRHO)) * 1E+5  # in pcm

        plt.figure(figsize=(8, 4))
        plt.plot(time_step, dRHO_time)
        plt.title("dRHO vs. Time")
        plt.xlabel("Time (s)")
        plt.ylabel(r"$\delta \rho$ (pcm)")
        plt.grid(True)
        plt.savefig(f'{output_dir}/{case_name}_PK_COMPONENTS/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_dRHO_time.png')

    elif geom_type =='3D rectangular':
        output_dir = f'OUTPUTS/{case_name}'
        x = globals().get("x")
        y = globals().get("y")
        z = globals().get("z")
        dx = globals().get("dx")
        dy = globals().get("dy")
        dz = globals().get("dz")
        I_max = globals().get("I_max")
        J_max = globals().get("J_max")
        K_max = globals().get("K_max")
        N = globals().get("N")
        group = globals().get("group")
        D = globals().get("D")
        TOT = globals().get("TOT")
        SIGS_reshaped = globals().get("SIGS_reshaped")
        chi = globals().get("chi")
        NUFIS = globals().get("NUFIS")
        BC = globals().get("BC")

        # Forward Simulation
        solver_type = 'forward'
        os.makedirs(f'{output_dir}/{case_name}_PK_COMPONENTS/{case_name}_{solver_type.upper()}', exist_ok=True)
        conv = convert_index_3D_rect(D, I_max, J_max, K_max)
        conv_array = np.array(conv)

        matrix_builder = MatrixBuilderForward3DRect(group, N, conv, TOT, SIGS_reshaped, BC, dx, dy, dz, D, chi, NUFIS)
        M, F_FORWARD = matrix_builder.build_forward_matrices()

        solver = SolverFactory.get_solver_power3DRect(solver_type, group, N, conv, M, F_FORWARD, dx, dy, dz, precond, tol=1E-10)
        keff, PHI_temp = solver.solve()
        PHI, PHI_reshaped, PHI_reshaped_plot = PostProcessor.postprocess_power3DRect(PHI_temp, conv, group, N, I_max, J_max, K_max)

        output = {"keff": keff.real}
        for g in range(len(PHI_reshaped)):
            phi_groupname = f'PHI{g + 1}_{solver_type.upper()}'
            output[phi_groupname] = [val.real for val in PHI_reshaped[g]]

        with open(f'{output_dir}/{case_name}_PK_COMPONENTS/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_output.json', 'w') as json_file:
            json.dump(output, json_file, indent=4)

        PHI_temp_reshaped = np.reshape(PHI_temp, (group, max(conv)))

        # Adjoint Simulation
        solver_type = 'adjoint'
        os.makedirs(f'{output_dir}/{case_name}_PK_COMPONENTS/{case_name}_{solver_type.upper()}', exist_ok=True)
        conv = convert_index_3D_rect(D, I_max, J_max, K_max)
        conv_array = np.array(conv)

        matrix_builder = MatrixBuilderAdjoint3DRect(group, N, conv, TOT, SIGS_reshaped, BC, dx, dy, dz, D, chi, NUFIS)
        M, F_ADJOINT = matrix_builder.build_adjoint_matrices()

        solver = SolverFactory.get_solver_power3DRect(solver_type, group, N, conv, M, F_ADJOINT, dx, dy, dz, precond, tol=1E-10)
        keff, PHI_ADJ_temp = solver.solve()
        PHI_ADJ, PHI_ADJ_reshaped, PHI_ADJ_reshaped_plot = PostProcessor.postprocess_power3DRect(PHI_ADJ_temp, conv, group, N, I_max, J_max, K_max)

        output = {"keff": keff.real}
        for g in range(len(PHI_ADJ_reshaped)):
            phi_groupname = f'PHI{g + 1}_{solver_type.upper()}'
            output[phi_groupname] = [val.real for val in PHI_ADJ_reshaped[g]]

        with open(f'{output_dir}/{case_name}_PK_COMPONENTS/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_output.json', 'w') as json_file:
            json.dump(output, json_file, indent=4)

        PHI_ADJ_temp_reshaped = np.reshape(PHI_ADJ_temp, (group, max(conv)))

        # Noise Simulation
        solver_type = 'noise'
        v = globals().get("v")
        Beff = globals().get("Beff")
        omega = globals().get("omega")
        l = globals().get("l")
        dTOT = globals().get("dTOT")
        dSIGS_reshaped = globals().get("dSIGS_reshaped")
        dNUFIS = globals().get("dNUFIS")

        os.makedirs(f'{output_dir}/{case_name}_PK_COMPONENTS/{case_name}_{solver_type.upper()}', exist_ok=True)
        conv = convert_index_3D_rect(D, I_max, J_max, K_max)
        conv_array = np.array(conv)

        with open(f'{output_dir}/{case_name}_PK_COMPONENTS/{case_name}_FORWARD/{case_name}_FORWARD_output.json', 'r') as json_file:
            forward_output = json.load(json_file)
        keff = forward_output["keff"]
        PHI_all = []
        for i in range(group):
            phi_key = f"PHI{i+1}_FORWARD"
            PHI_all.append(forward_output[phi_key])

        PHI = np.zeros(max(conv) * group)
        for g in range(group):
            PHI_indices = g * max(conv) + (conv_array - 1)
            PHI[PHI_indices] = PHI_all[g]

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
    
        with open(f'{output_dir}/{case_name}_PK_COMPONENTS/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_output.json', 'w') as json_file:
            json.dump(output, json_file, indent=4)

        dPHI_temp_reshaped = np.reshape(dPHI_temp, (group, max(conv)))

        v1_PP = v1[0][0][0]
        v2_PP = v2[0][0][0]
        v_new = [v1_PP, v2_PP]

        NUMER = 0
        DENOM = 0
        for g in range(group):
            NUMER += trapezoid((1/v_new[g]) * (PHI_ADJ_temp_reshaped[g] * dPHI_temp_reshaped[g]), dx=dx*dy*dz, axis = 0)
            DENOM += trapezoid((1/v_new[g]) * (PHI_ADJ_temp_reshaped[g] * PHI_temp_reshaped[g]), dx=dx*dy*dz, axis = 0)

        dPOWER = (NUMER / DENOM)

        S_NUMER = dS.dot(PHI)
        S_NUMER_reshaped = np.reshape(S_NUMER, (group, max(conv)))
        S_DENOM = F_FORWARD.dot(PHI)
        S_DENOM_reshaped = np.reshape(S_DENOM, (group, max(conv)))
        S_NUMER_reshaped = np.nan_to_num(S_NUMER_reshaped, nan=0)
        S_DENOM_reshaped = np.nan_to_num(S_DENOM_reshaped, nan=0)

        NUMER = 0
        DENOM = 0
        for g in range(group):
            NUMER += trapezoid(PHI_ADJ_temp_reshaped[g] * S_NUMER_reshaped[g], dx=dx*dy*dz, axis = 0)
            DENOM += trapezoid(PHI_ADJ_temp_reshaped[g] * S_DENOM_reshaped[g], dx=dx*dy*dz, axis = 0)

        dRHO = (NUMER / DENOM)

        dPHI_pk_temp = dPOWER * PHI
        dPHI_spatial_temp = dPHI_temp - dPHI_pk_temp

        dPHI_pk, dPHI_pk_reshaped, dPHI_pk_reshaped_plot = PostProcessor.postprocess_fixed3DRect(dPHI_pk_temp, conv, group, N, I_max, J_max, K_max)
        output = {}
        for g in range(len(dPHI_pk_reshaped)):
            dPHI_pk_groupname = f'dPHI{g + 1}_pk'
            dPHI_pk_list = [{"real": x.real, "imaginary": x.imag} for x in dPHI_pk_reshaped[g]]
            output[dPHI_pk_groupname] = dPHI_pk_list
        with open(f'{output_dir}/{case_name}_PK_COMPONENTS/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_pk_output.json', 'w') as json_file:
            json.dump(output, json_file, indent=4)

        dPHI_spatial, dPHI_spatial_reshaped, dPHI_spatial_reshaped_plot = PostProcessor.postprocess_fixed3DRect(dPHI_spatial_temp, conv, group, N, I_max, J_max, K_max)
        output = {}
        for g in range(len(dPHI_spatial_reshaped)):
            dPHI_spatial_groupname = f'dPHI{g + 1}_spatial'
            dPHI_spatial_list = [{"real": x.real, "imaginary": x.imag} for x in dPHI_spatial_reshaped[g]]
            output[dPHI_spatial_groupname] = dPHI_spatial_list
        with open(f'{output_dir}/{case_name}_PK_COMPONENTS/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_spatial_output.json', 'w') as json_file:
            json.dump(output, json_file, indent=4)

        dPHI_pk = np.zeros(group * N, dtype=complex)
        dPHI_spatial = np.zeros(group * N, dtype=complex)
        conv_array = np.array(conv)
        non_zero_indices = np.nonzero(conv)[0]
        phi_temp_indices = conv_array[non_zero_indices] - 1

        for g in range(group):
            dPHI_temp_start = g * max(conv)
            dPHI_pk[g * N + non_zero_indices] = dPHI_pk_temp[dPHI_temp_start + phi_temp_indices]
            dPHI_spatial[g * N + non_zero_indices] = dPHI_spatial_temp[dPHI_temp_start + phi_temp_indices]

            for n in range(N):
                if conv[n] == 0:
                    dPHI_pk[g * N + n] = np.nan
                    dPHI_spatial[g * N + n] = np.nan

        dPHI_pk_plot = dPHI_pk.reshape(group, K_max, J_max, I_max)
        dPHI_spatial_plot = dPHI_spatial.reshape(group, K_max, J_max, I_max)

        for g in range(group):
            image_mag_files = []
            image_phase_files = []
            for k in range(K_max):
                filename_mag = plot_heatmap_3D(dPHI_reshaped_plot[g, k, :, :], g+1, k+1, x, y, cmap='viridis', varname='dPHI', title=f'2D Plot of dPHI{g+1}, Z={k+1}, Magnitude', output_dir=f'{output_dir}/{case_name}_PK_COMPONENTS', case_name=case_name, process_data='magnitude', solve=solver_type.upper())
                filename_phase = plot_heatmap_3D(dPHI_reshaped_plot[g, k, :, :], g+1, k+1, x, y, cmap='viridis', varname='dPHI', title=f'2D Plot of dPHI{g+1}, Z={k+1}, Phase', output_dir=f'{output_dir}/{case_name}_PK_COMPONENTS', case_name=case_name, process_data='phase', solve=solver_type.upper())
                image_mag_files.append(filename_mag)
                image_phase_files.append(filename_phase)

            # Create a GIF from the saved images
            gif_filename_mag = f'{output_dir}/{case_name}_PK_COMPONENTS/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_dPHI_animation_magnitude_G{g+1}.gif'
            gif_filename_phase = f'{output_dir}/{case_name}_PK_COMPONENTS/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_dPHI_animation_phase_G{g+1}.gif'

            # Open images and save as GIF
            images_mag = [Image.open(img) for img in image_mag_files]
            images_mag[0].save(gif_filename_mag, save_all=True, append_images=images_mag[1:], duration=300, loop=0)
            print(f"GIF saved as {gif_filename_mag}")

            images_phase = [Image.open(img) for img in image_phase_files]
            images_phase[0].save(gif_filename_phase, save_all=True, append_images=images_phase[1:], duration=300, loop=0)
            print(f"GIF saved as {gif_filename_phase}")

        for g in range(group):
            image_mag_files = []
            image_phase_files = []
            for k in range(K_max):
                filename_mag = plot_heatmap_3D(dPHI_pk_plot[g, k, :, :], g+1, k+1, x, y, cmap='viridis', varname='dPHI_pk', title=f'2D Plot of dPHI{g+1}_pk, Z={k+1}, Magnitude', output_dir=f'{output_dir}/{case_name}_PK_COMPONENTS', case_name=case_name, process_data='magnitude', solve=solver_type.upper())
                filename_phase = plot_heatmap_3D(dPHI_pk_plot[g, k, :, :], g+1, k+1, x, y, cmap='viridis', varname='dPHI_pk', title=f'2D Plot of dPHI{g+1}_pk, Z={k+1}, Phase', output_dir=f'{output_dir}/{case_name}_PK_COMPONENTS', case_name=case_name, process_data='phase', solve=solver_type.upper())
                image_mag_files.append(filename_mag)
                image_phase_files.append(filename_phase)

            # Create a GIF from the saved images
            gif_filename_mag = f'{output_dir}/{case_name}_PK_COMPONENTS/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_dPHI_pk_animation_magnitude_G{g+1}.gif'
            gif_filename_phase = f'{output_dir}/{case_name}_PK_COMPONENTS/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_dPHI_pk_animation_phase_G{g+1}.gif'

            # Open images and save as GIF
            images_mag = [Image.open(img) for img in image_mag_files]
            images_mag[0].save(gif_filename_mag, save_all=True, append_images=images_mag[1:], duration=300, loop=0)
            print(f"GIF saved as {gif_filename_mag}")

            images_phase = [Image.open(img) for img in image_phase_files]
            images_phase[0].save(gif_filename_phase, save_all=True, append_images=images_phase[1:], duration=300, loop=0)
            print(f"GIF saved as {gif_filename_phase}")

        for g in range(group):
            image_mag_files = []
            image_phase_files = []
            for k in range(K_max):
                filename_mag = plot_heatmap_3D(dPHI_spatial_plot[g, k, :, :], g+1, k+1, x, y, cmap='viridis', varname='dPHI_spatial', title=f'2D Plot of dPHI{g+1}_spatial, Z={k+1}, Magnitude', output_dir=f'{output_dir}/{case_name}_PK_COMPONENTS', case_name=case_name, process_data='magnitude', solve=solver_type.upper())
                filename_phase = plot_heatmap_3D(dPHI_spatial_plot[g, k, :, :], g+1, k+1, x, y, cmap='viridis', varname='dPHI_spatial', title=f'2D Plot of dPHI{g+1}_spatial, Z={k+1}, Phase', output_dir=f'{output_dir}/{case_name}_PK_COMPONENTS', case_name=case_name, process_data='phase', solve=solver_type.upper())
                image_mag_files.append(filename_mag)
                image_phase_files.append(filename_phase)

            # Create a GIF from the saved images
            gif_filename_mag = f'{output_dir}/{case_name}_PK_COMPONENTS/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_dPHI_spatial_animation_magnitude_G{g+1}.gif'
            gif_filename_phase = f'{output_dir}/{case_name}_PK_COMPONENTS/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_dPHI_spatial_animation_phase_G{g+1}.gif'

            # Open images and save as GIF
            images_mag = [Image.open(img) for img in image_mag_files]
            images_mag[0].save(gif_filename_mag, save_all=True, append_images=images_mag[1:], duration=300, loop=0)
            print(f"GIF saved as {gif_filename_mag}")

            images_phase = [Image.open(img) for img in image_phase_files]
            images_phase[0].save(gif_filename_phase, save_all=True, append_images=images_phase[1:], duration=300, loop=0)
            print(f"GIF saved as {gif_filename_phase}")

        time_step = np.linspace(0,10, 1001)
        dRHO_time = np.abs(dRHO) * np.cos(2 * np.pi * f * time_step + np.angle(dRHO)) * 1E+5  # in pcm

        plt.figure(figsize=(8, 4))
        plt.plot(time_step, dRHO_time)
        plt.title("dRHO vs. Time")
        plt.xlabel("Time (s)")
        plt.ylabel(r"$\delta \rho$ (pcm)")
        plt.grid(True)
        plt.savefig(f'{output_dir}/{case_name}_PK_COMPONENTS/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_dRHO_time.png')

    elif geom_type =='3D triangular':
        h = globals().get("h")
        dz = globals().get("dz")
        s = globals().get("s")
        N_hexx = globals().get("N_hexx")
        level = globals().get("level")
        I_max = globals().get("I_max")
        J_max = globals().get("J_max")
        K_max = globals().get("K_max")
        N = globals().get("N")
        group = globals().get("group")
        D = globals().get("D")
        TOT = globals().get("TOT")
        SIGS_reshaped = globals().get("SIGS_reshaped")
        chi = globals().get("chi")
        NUFIS = globals().get("NUFIS")
        BC = globals().get("BC")
        input_name = globals().get("input_name")
        output_dir = f'OUTPUTS/{input_name}'

        # Forward Simulation
        solver_type = 'forward'
        os.makedirs(f'{output_dir}/{case_name}_PK_COMPONENTS/{case_name}_{solver_type.upper()}', exist_ok=True)
        conv_hexx = convert_3D_hexx(K_max, J_max, I_max, D)
        conv_tri, conv_hexx_ext = convert_3D_tri(K_max, J_max, I_max, conv_hexx, level)
        conv_tri_array = np.array(conv_tri)
        conv_neighbor_2D, conv_neighbor_3D, tri_indices, x, y, all_triangles = calculate_neighbors_3D(s, I_max, J_max, K_max, conv_hexx, level)

        matrix_builder = MatrixBuilderForward3DHexx(group, I_max, J_max, K_max, conv_tri, conv_neighbor_3D, TOT, SIGS_reshaped, BC, h, dz, level, D, chi, NUFIS)
        M, F_FORWARD = matrix_builder.build_forward_matrices()

        solver = SolverFactory.get_solver_power3DHexx(solver_type, group, conv_tri, M, F_FORWARD, h, dz, precond, tol=1E-10)
        keff, PHI_temp = solver.solve()
        PHI, PHI_reshaped, PHI_temp_reshaped = PostProcessor.postprocess_power3DHexx(PHI_temp, conv_tri, group, N_hexx, K_max, tri_indices)

        PHI_temp_reshaped_new = np.reshape(PHI_temp, (group, max(conv_tri)))
        PHI_temp_reshaped_new = np.nan_to_num(PHI_temp_reshaped_new, nan=0)

        output = {"keff": keff.real}
        for g in range(len(PHI_reshaped)):
            phi_groupname = f'PHI{g + 1}_{solver_type.upper()}'
            output[phi_groupname] = [val.real for val in PHI_reshaped[g]]

        with open(f'{output_dir}/{case_name}_PK_COMPONENTS/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_output.json', 'w') as json_file:
            json.dump(output, json_file, indent=4)

        # Adjoint Simulation
        solver_type = 'adjoint'
        os.makedirs(f'{output_dir}/{case_name}_PK_COMPONENTS/{case_name}_{solver_type.upper()}', exist_ok=True)

        matrix_builder = MatrixBuilderAdjoint3DHexx(group, I_max, J_max, K_max, conv_tri, conv_neighbor_3D, TOT, SIGS_reshaped, BC, h, dz, level, D, chi, NUFIS)
        M, F_ADJOINT = matrix_builder.build_adjoint_matrices()

        solver = SolverFactory.get_solver_power3DHexx(solver_type, group, conv_tri, M, F_ADJOINT, h, dz, precond, tol=1E-10)
        keff, PHI_ADJ_temp = solver.solve()
        PHI_ADJ, PHI_ADJ_reshaped, PHI_ADJ_temp_reshaped = PostProcessor.postprocess_power3DHexx(PHI_ADJ_temp, conv_tri, group, N_hexx, K_max, tri_indices)

        PHI_ADJ_temp_reshaped_new = np.reshape(PHI_ADJ_temp, (group, max(conv_tri)))
        PHI_ADJ_temp_reshaped_new = np.nan_to_num(PHI_ADJ_temp_reshaped_new, nan=0)

        output = {"keff": keff.real}
        for g in range(len(PHI_ADJ_reshaped)):
            phi_groupname = f'PHI{g + 1}_{solver_type.upper()}'
            output[phi_groupname] = [val.real for val in PHI_ADJ_reshaped[g]]

        with open(f'{output_dir}/{case_name}_PK_COMPONENTS/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_output.json', 'w') as json_file:
            json.dump(output, json_file, indent=4)

        # Noise Simulation
        solver_type = 'noise'
        v = globals().get("v")
        Beff = globals().get("Beff")
        omega = globals().get("omega")
        l = globals().get("l")
        dTOT = globals().get("dTOT")
        dSIGS_reshaped = globals().get("dSIGS_reshaped")
        dNUFIS = globals().get("dNUFIS")
        noise_section = globals().get("noise_section")
        os.makedirs(f'{output_dir}/{case_name}_PK_COMPONENTS/{case_name}_{solver_type.upper()}', exist_ok=True)

        with open(f'{output_dir}/{case_name}_PK_COMPONENTS/{case_name}_FORWARD/{case_name}_FORWARD_output.json', 'r') as json_file:
            forward_output = json.load(json_file)
        keff = forward_output["keff"]
        PHI_all = []
        for i in range(group):
            phi_key = f"PHI{i+1}_FORWARD"
            PHI_all.append(forward_output[phi_key])

        PHI = np.zeros(max(conv_tri) * group)
        for g in range(group):
            PHI_indices = g * max(conv_tri) + (conv_tri_array - 1)
            PHI[PHI_indices] = PHI_all[g]

        dTOT_hexx = expand_XS_hexx_3D(group, K_max, J_max, I_max, dTOT, level)
        dSIGS_hexx = expand_SIGS_hexx_3D(group, K_max, J_max, I_max, dSIGS_reshaped, level)
        chi_hexx = expand_XS_hexx_3D(group, K_max, J_max, I_max, chi, level)
        dNUFIS_hexx = expand_XS_hexx_3D(group, K_max, J_max, I_max, dNUFIS, level)

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

        hex_centers, hex_vertices = generate_pointy_hex_grid(s, I_max, J_max)
        triangle_neighbors_global = find_triangle_neighbors_2D(all_triangles, precision=6)

        if type_noise == 'FXV':
            dTOT_hexx, dNUFIS_hexx = XS3D_FXV(level, group, s, K_max, J_max, I_max, dTOT, dNUFIS, fav_strength, diff_X_ABS, diff_X_NUFIS)
        elif type_noise == 'FAV':
            dTOT_hexx, dNUFIS_hexx = XS3D_FAV(level, group, s, K_max, J_max, I_max, dTOT, dNUFIS, fav_strength, diff_X_ABS, diff_X_NUFIS)

        matrix_builder = MatrixBuilderNoise3DHexx(group, I_max, J_max, K_max, N_hexx, conv_tri, conv_neighbor_3D, TOT, SIGS_reshaped, BC, h, dz, level, D, chi, NUFIS, keff, v, Beff, omega, l, dTOT_hexx, dSIGS_hexx, chi_hexx, dNUFIS_hexx, noise_section, type_noise)
        M, dS = matrix_builder.build_noise_matrices()

        solver = SolverFactory.get_solver_fixed3DHexx(solver_type, group, conv_tri, M, dS, PHI, precond, tol=1e-10)

        dPHI_temp = solver.solve()
        dPHI, dPHI_reshaped, dPHI_temp_reshaped = PostProcessor.postprocess_fixed3DHexx(dPHI_temp, conv_tri, group, N_hexx, K_max, tri_indices)
        dPHI_temp_reshaped_new = np.reshape(dPHI_temp, (group, max(conv_tri)))
        dPHI_temp_reshaped_new = np.nan_to_num(dPHI_temp_reshaped_new, nan=0)

        output = {}
        for g in range(len(dPHI_reshaped)):
            dPHI_groupname = f'dPHI{g + 1}'
            dPHI_list = [{"real": x.real, "imaginary": x.imag} for x in dPHI_reshaped[g]]
            output[dPHI_groupname] = dPHI_list
    
        with open(f'{output_dir}/{case_name}_PK_COMPONENTS/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_output.json', 'w') as json_file:
            json.dump(output, json_file, indent=4)

        v1_PP = v1
        v2_PP = v2
        v_new = [v1_PP, v2_PP]

        NUMER = 0
        DENOM = 0

        for g in range(group):
            NUMER += trapezoid((1/v_new[g]) * (PHI_ADJ_temp_reshaped_new[g] * dPHI_temp_reshaped_new[g]), dx=h**2/4*np.sqrt(3)*dz, axis = 0)
            DENOM += trapezoid((1/v_new[g]) * (PHI_ADJ_temp_reshaped_new[g] * PHI_temp_reshaped_new[g]), dx=h**2/4*np.sqrt(3)*dz, axis = 0)
        dPOWER = (NUMER / DENOM)

        S_NUMER = dS.dot(PHI)
        S_NUMER_reshaped = np.reshape(S_NUMER, (group, max(conv_tri)))
        S_DENOM = F_FORWARD.dot(PHI)
        S_DENOM_reshaped = np.reshape(S_DENOM, (group, max(conv_tri)))
        S_NUMER_reshaped = np.nan_to_num(S_NUMER_reshaped, nan=0)
        S_DENOM_reshaped = np.nan_to_num(S_DENOM_reshaped, nan=0)

        NUMER = 0
        DENOM = 0
        for g in range(group):
            NUMER += trapezoid(PHI_ADJ_temp_reshaped_new[g] * S_NUMER_reshaped[g], dx=h**2/4*np.sqrt(3)*dz, axis = 0)
            DENOM += trapezoid(PHI_ADJ_temp_reshaped_new[g] * S_DENOM_reshaped[g], dx=h**2/4*np.sqrt(3)*dz, axis = 0)

        dRHO = (NUMER / DENOM)

        dPHI_pk_temp = dPOWER * PHI
        dPHI_spatial_temp = dPHI_temp - dPHI_pk_temp

        dPHI_pk, dPHI_pk_reshaped, dPHI_pk_temp_reshaped = PostProcessor.postprocess_fixed3DHexx(dPHI_pk_temp, conv_tri, group, N_hexx, K_max, tri_indices)
        output = {}
        for g in range(len(dPHI_pk_reshaped)):
            dPHI_pk_groupname = f'dPHI{g + 1}_pk'
            dPHI_pk_list = [{"real": x.real, "imaginary": x.imag} for x in dPHI_pk_reshaped[g]]
            output[dPHI_pk_groupname] = dPHI_pk_list

        with open(f'{output_dir}/{case_name}_PK_COMPONENTS/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_pk_output.json', 'w') as json_file:
            json.dump(output, json_file, indent=4)

        dPHI_spatial, dPHI_spatial_reshaped, dPHI_spatial_temp_reshaped = PostProcessor.postprocess_fixed3DHexx(dPHI_spatial_temp, conv_tri, group, N_hexx, K_max, tri_indices)
        output = {}
        for g in range(len(dPHI_spatial_reshaped)):
            dPHI_spatial_groupname = f'dPHI{g + 1}_spatial'
            dPHI_spatial_list = [{"real": x.real, "imaginary": x.imag} for x in dPHI_spatial_reshaped[g]]
            output[dPHI_spatial_groupname] = dPHI_spatial_list

        with open(f'{output_dir}/{case_name}_PK_COMPONENTS/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_spatial_output.json', 'w') as json_file:
            json.dump(output, json_file, indent=4)

        dPHI_pk_plot = dPHI_pk_temp.reshape(group, K_max, len(tri_indices))
        dPHI_spatial_plot = dPHI_spatial_temp.reshape(group, K_max, len(tri_indices))

        for g in range(group):
            image_files_mag = []
            image_files_phase = []
            for k in range(K_max):
                filename_dPHI_mag = plot_triangular_3D(dPHI_temp_reshaped[g][k], x, y, k+1, tri_indices, g+1, cmap='viridis', varname='dPHI', title=f'2D Plot of dPHI{g+1}, Z{k+1} Hexx Magnitude', case_name=case_name, output_dir=f'{output_dir}/{case_name}_PK_COMPONENTS', solve=solver_type.upper(), process_data="magnitude")
                filename_dPHI_phase = plot_triangular_3D(dPHI_temp_reshaped[g][k], x, y, k+1, tri_indices, g+1, cmap='viridis', varname='dPHI', title=f'2D Plot of dPHI{g+1}, Z{k+1} Hexx Phase', case_name=case_name, output_dir=f'{output_dir}/{case_name}_PK_COMPONENTS', solve=solver_type.upper(), process_data="phase")
                image_files_mag.append(filename_dPHI_mag)
                image_files_phase.append(filename_dPHI_phase)

            # Create a GIF from the saved images
            gif_filename_dPHI_mag = f'{output_dir}/{case_name}_PK_COMPONENTS/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_dPHI_animation_G{g+1}_magnitude.gif'
            gif_filename_dPHI_phase = f'{output_dir}/{case_name}_PK_COMPONENTS/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_dPHI_animation_G{g+1}_phase.gif'

            # Open images and save as GIF
            images_dPHI_mag = [Image.open(img) for img in image_files_mag]
            images_dPHI_phase = [Image.open(img) for img in image_files_phase]
            images_dPHI_mag[0].save(gif_filename_dPHI_mag, save_all=True, append_images=images_dPHI_mag[1:], duration=300, loop=0)
            images_dPHI_phase[0].save(gif_filename_dPHI_phase, save_all=True, append_images=images_dPHI_phase[1:], duration=300, loop=0)
            print(f"GIF saved as {gif_filename_dPHI_mag}")
            print(f"GIF saved as {gif_filename_dPHI_phase}")

        for g in range(group):
            image_files_mag = []
            image_files_phase = []
            for k in range(K_max):
                filename_dPHI_mag = plot_triangular_3D(dPHI_pk_plot[g][k], x, y, k+1, tri_indices, g+1, cmap='viridis', varname='dPHI_pk', title=f'2D Plot of dPHI{g+1}_pk, Z{k+1} Hexx Magnitude', case_name=case_name, output_dir=f'{output_dir}/{case_name}_PK_COMPONENTS', solve=solver_type.upper(), process_data="magnitude")
                filename_dPHI_phase = plot_triangular_3D(dPHI_pk_plot[g][k], x, y, k+1, tri_indices, g+1, cmap='viridis', varname='dPHI_pk', title=f'2D Plot of dPHI{g+1}_pk, Z{k+1} Hexx Phase', case_name=case_name, output_dir=f'{output_dir}/{case_name}_PK_COMPONENTS', solve=solver_type.upper(), process_data="phase")
                image_files_mag.append(filename_dPHI_mag)
                image_files_phase.append(filename_dPHI_phase)

            # Create a GIF from the saved images
            gif_filename_dPHI_mag = f'{output_dir}/{case_name}_PK_COMPONENTS/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_dPHI_pk_animation_G{g+1}_magnitude.gif'
            gif_filename_dPHI_phase = f'{output_dir}/{case_name}_PK_COMPONENTS/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_dPHI_pk_animation_G{g+1}_phase.gif'

            # Open images and save as GIF
            images_dPHI_mag = [Image.open(img) for img in image_files_mag]
            images_dPHI_phase = [Image.open(img) for img in image_files_phase]
            images_dPHI_mag[0].save(gif_filename_dPHI_mag, save_all=True, append_images=images_dPHI_mag[1:], duration=300, loop=0)
            images_dPHI_phase[0].save(gif_filename_dPHI_phase, save_all=True, append_images=images_dPHI_phase[1:], duration=300, loop=0)
            print(f"GIF saved as {gif_filename_dPHI_mag}")
            print(f"GIF saved as {gif_filename_dPHI_phase}")

        for g in range(group):
            image_files_mag = []
            image_files_phase = []
            for k in range(K_max):
                filename_dPHI_mag = plot_triangular_3D(dPHI_spatial_plot[g][k], x, y, k+1, tri_indices, g+1, cmap='viridis', varname='dPHI_spatial', title=f'2D Plot of dPHI{g+1}_spatial, Z{k+1} Hexx Magnitude', case_name=case_name, output_dir=f'{output_dir}/{case_name}_PK_COMPONENTS', solve=solver_type.upper(), process_data="magnitude")
                filename_dPHI_phase = plot_triangular_3D(dPHI_spatial_plot[g][k], x, y, k+1, tri_indices, g+1, cmap='viridis', varname='dPHI_spatial', title=f'2D Plot of dPHI{g+1}_spatial, Z{k+1} Hexx Phase', case_name=case_name, output_dir=f'{output_dir}/{case_name}_PK_COMPONENTS', solve=solver_type.upper(), process_data="phase")
                image_files_mag.append(filename_dPHI_mag)
                image_files_phase.append(filename_dPHI_phase)

            # Create a GIF from the saved images
            gif_filename_dPHI_mag = f'{output_dir}/{case_name}_PK_COMPONENTS/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_dPHI_spatial_animation_G{g+1}_magnitude.gif'
            gif_filename_dPHI_phase = f'{output_dir}/{case_name}_PK_COMPONENTS/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_dPHI_spatial_animation_G{g+1}_phase.gif'

            # Open images and save as GIF
            images_dPHI_mag = [Image.open(img) for img in image_files_mag]
            images_dPHI_phase = [Image.open(img) for img in image_files_phase]
            images_dPHI_mag[0].save(gif_filename_dPHI_mag, save_all=True, append_images=images_dPHI_mag[1:], duration=300, loop=0)
            images_dPHI_phase[0].save(gif_filename_dPHI_phase, save_all=True, append_images=images_dPHI_phase[1:], duration=300, loop=0)
            print(f"GIF saved as {gif_filename_dPHI_mag}")
            print(f"GIF saved as {gif_filename_dPHI_phase}")

        time_step = np.linspace(0,10, 1001)
        dRHO_time = np.abs(dRHO) * np.cos(2 * np.pi * f * time_step + np.angle(dRHO)) * 1E+5  # in pcm

        plt.figure(figsize=(8, 4))
        plt.plot(time_step, dRHO_time)
        plt.title("dRHO vs. Time")
        plt.xlabel("Time (s)")
        plt.ylabel(r"$\delta \rho$ (pcm)")
        plt.grid(True)
        plt.savefig(f'{output_dir}/{case_name}_PK_COMPONENTS/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_dRHO_time.png')

if __name__ == "__main__":
    main()