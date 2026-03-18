import numpy as np
import json
import time
import os
import sys

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

#######################################################################################################
# FOR TEST PURPOSES
#from INPUTS.OBJECTIVES1_TEST01_1DMG_CSTest03 import *
#from INPUTS.OBJECTIVES1_TEST02_2DMG_C3_VandV import *
#from INPUTS.OBJECTIVES1_TEST04_2DTriMG_HOMOG_VandV import *
#from INPUTS.OBJECTIVES1_TEST09_3DTriMG_HTTR import *
from INPUTS.OBJECTIVES1_TEST10_3DMG_Langenbuch import *

#######################################################################################################
#from INPUTS.OBJECTIVES1_TEST03_2DMG_BIBLIS_VandV import *
#from INPUTS.OBJECTIVES1_TEST05_2DTriMG_VVER400_VandV import *
#from INPUTS.OBJECTIVES1_TEST06_3DMG_CSTest09_VandV_new import *
#from INPUTS.OBJECTIVES1_TEST07_3DTriMG_VVER400_VandV import *
#from INPUTS.OBJECTIVES1_TEST08_2DTriMG_HTTR2G_VandV import *
#from INPUTS.OBJECTIVES1_TEST09_3DTriMG_HTTR import *
#from INPUTS.OBJECTIVES1_TEST10_3DMG_Langenbuch import *

#from INPUTS.OBJECTIVES3_TEST01_2DMG_BIBLIS_AVS import *
#from INPUTS.OBJECTIVES3_TEST01_2DMG_BIBLIS_CENTER_AVS import *
#from INPUTS.OBJECTIVES3_TEST02_2DMG_BIBLIS_FAV import *
#from INPUTS.OBJECTIVES3_TEST03_2DTriMG_HTTR2G_AVS import *
#from INPUTS.OBJECTIVES3_TEST04_2DTriMG_HTTR2G_FAV import *
#from INPUTS.OBJECTIVES3_TEST05_3DMG_CSTest09_AVS import *
#from INPUTS.OBJECTIVES3_TEST06_3DMG_CSTest09_FAV import *
#from INPUTS.OBJECTIVES3_TEST07_3DTriMG_HTTR_AVS import *
#from INPUTS.OBJECTIVES3_TEST08_3DTriMG_HTTR_FAV import *
#from INPUTS.OBJECTIVES3_TEST09_3DMG_Langenbuch_AVS import *

#from INPUTS.TASK3_TEST04_2DTriMG_HTTR2G_FAV import *
#from INPUTS.OBJECTIVES3_TEST04a_2DTriMG_HTTR2G_FAV_TRIAL import *

#from INPUTS.NPRE555C1_recreate import *

#######################################################################################################
#solver_type = 'forward'
solver_type = 'adjoint'
#solver_type = 'noise'

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

        Utils.create_directories(solver_type, output_dir, case_name)
        if solver_type in ['forward', 'adjoint']:
            if solver_type == 'forward':
                matrix_builder = MatrixBuilderForward1D(group, N, TOT, SIGS, BC, dx, D, chi, NUFIS)
                M, F = matrix_builder.build_forward_matrices()
            elif solver_type == 'adjoint':
                matrix_builder = MatrixBuilderAdjoint1D(group, N, TOT, SIGS, BC, dx, D, chi, NUFIS)
                M, F = matrix_builder.build_adjoint_matrices()

            solver = SolverFactory.get_solver_power1D(solver_type, group, N, M, F, x, precond, tol=1E-10)
            keff, PHI = solver.solve()
            PHI_reshaped = np.reshape(PHI, (group, N))
            PostProcessor.save_output_power1D(output_dir, case_name, keff, PHI_reshaped, solver_type)
            for g in range(group):
                Utils.plot_1D_power(solver_type, PHI_reshaped[g], x, g, output_dir=output_dir, varname=f'PHI', case_name=case_name, title=f'1D Plot of PHI{g+1}')
        elif solver_type == 'noise':
            v = globals().get("v")
            Beff = globals().get("Beff")
            omega = globals().get("omega")
            l = globals().get("l")
            dTOT = globals().get("dTOT")
            dSIGS = globals().get("dSIGS")
            dNUFIS = globals().get("dNUFIS")
            dSOURCE = globals().get("dSOURCE")
            # Load data from JSON file
            with open(f'{output_dir}/{case_name}_FORWARD/{case_name}_FORWARD_output.json', 'r') as json_file:
                forward_output = json.load(json_file)

            # Access keff and PHI from the loaded data
            keff = forward_output["keff"]
            PHI = []
            for i in range(group):
                phi_key = f"PHI{i+1}_FORWARD"
                PHI.extend(forward_output[phi_key])
            dSOURCE_new = [item for sublist in dSOURCE for item in sublist]

            matrix_builder = MatrixBuilderNoise1D(group, N, TOT, SIGS, BC, dx, D, chi, NUFIS, keff, v, Beff, omega, l, dTOT, dSIGS, dNUFIS)
            M, dS = matrix_builder.build_noise_matrices()

            solver = SolverFactory.get_solver_fixed1D(solver_type, group, N, M, dS, dSOURCE, PHI, precond, tol=1e-10)

            dPHI = solver.solve()
            dPHI_reshaped = np.reshape(dPHI, (group, N))
            PostProcessor.save_output_fixed1D(output_dir, case_name, dPHI_reshaped, solver_type)
            for g in range(group):
                Utils.plot_1D_fixed(solver_type, dPHI_reshaped[g], x, g, output_dir=output_dir, varname=f'dPHI', case_name=case_name, title=f'1D Plot of dPHI{g+1}')

            S_plot = dS.dot(PHI) + dSOURCE_new
            S_plot_reshaped = np.reshape(S_plot, (group, N))
            for g in range(group):
                Utils.plot_1D_fixed(solver_type, S_plot_reshaped[g], x, g, output_dir=output_dir, varname=f'S', case_name=case_name, title=f'1D Plot of S{g+1}')

    elif geom_type =='2D rectangular':
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

        output_dir = f'OUTPUTS/{case_name}'
        Utils.create_directories(solver_type, output_dir, case_name)
        conv = convert_index_2D_rect(D, I_max, J_max)
        conv_array = np.array(conv)
        if solver_type in ['forward', 'adjoint']:
            if solver_type == 'forward':
                matrix_builder = MatrixBuilderForward2DRect(group, N, conv, TOT, SIGS_reshaped, BC, dx, dy, D, chi, NUFIS)
                M, F = matrix_builder.build_forward_matrices()
            elif solver_type == 'adjoint':
                matrix_builder = MatrixBuilderAdjoint2DRect(group, N, conv, TOT, SIGS_reshaped, BC, dx, dy, D, chi, NUFIS)
                M, F = matrix_builder.build_adjoint_matrices()

            solver = SolverFactory.get_solver_power2DRect(solver_type, group, N, conv, M, F, dx, dy, precond, tol=1E-10)
            keff, phi_temp = solver.solve()

            PHI, PHI_reshaped, PHI_reshaped_plot = PostProcessor.postprocess_power2DRect(phi_temp, conv, group, N, I_max, J_max)
            PostProcessor.save_output_power2DRect(output_dir, case_name, keff, PHI_reshaped, solver_type)
            for g in range(group):
                Utils.plot_2D_rect_power(solver_type, PHI_reshaped_plot[g], x, y, g+1, cmap='viridis', output_dir=output_dir, varname=f'PHI', case_name=case_name, title=f'2D Plot of PHI{g+1}')
        elif solver_type == 'noise':
            v = globals().get("v")
            Beff = globals().get("Beff")
            omega = globals().get("omega")
            l = globals().get("l")
            dTOT = globals().get("dTOT")
            dSIGS_reshaped = globals().get("dSIGS_reshaped")
            dNUFIS = globals().get("dNUFIS")

            # Load data from JSON file
            with open(f'{output_dir}/{case_name}_FORWARD/{case_name}_FORWARD_output.json', 'r') as json_file:
                forward_output = json.load(json_file)

            # Access keff and PHI from the loaded data
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
            PostProcessor.save_output_fixed2DRect(output_dir, case_name, keff, dPHI_reshaped, solver_type)
            for g in range(group):
                Utils.plot_2D_rect_fixed(solver_type, dPHI_reshaped_plot[g], x, y, g+1, cmap='viridis', output_dir=output_dir, varname=f'dPHI', case_name=case_name, title=f'2D Plot of dPHI{g+1} Magnitude', process_data='magnitude')
                Utils.plot_2D_rect_fixed(solver_type, dPHI_reshaped_plot[g], x, y, g+1, cmap='viridis', output_dir=output_dir, varname=f'dPHI', case_name=case_name, title=f'2D Plot of dPHI{g+1} Phase', process_data='phase')

            S_temp_plot = dS.dot(PHI)
            S_plot = np.zeros(group * N, dtype=complex)
            conv_array = np.array(conv)
            non_zero_indices = np.nonzero(conv)[0]
            phi_temp_indices = conv_array[non_zero_indices] - 1

            for g in range(group):
                phi_temp_start = g * max(conv)
                S_plot[g * N + non_zero_indices] = S_temp_plot[phi_temp_start + phi_temp_indices]

            for g in range(group):
                for n in range(N):
                    if conv[n] == 0:
                        S_plot[g * N + n] = np.nan
        
            S_reshaped_plot = np.reshape(S_plot, (group, I_max, J_max))
            for g in range(group):
                Utils.plot_2D_rect_fixed(solver_type, S_reshaped_plot[g], x, y, g+1, cmap='viridis', output_dir=output_dir, varname=f'S', case_name=case_name, title=f'2D Plot of S{g+1} Magnitude', process_data='magnitude')
                Utils.plot_2D_rect_fixed(solver_type, S_reshaped_plot[g], x, y, g+1, cmap='viridis', output_dir=output_dir, varname=f'S', case_name=case_name, title=f'2D Plot of S{g+1} Phase', process_data='phase')

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
        Utils.create_directories(solver_type, output_dir, case_name)
        conv_hexx = convert_2D_hexx(I_max, J_max, D)
        conv_tri, conv_hexx_ext = convert_2D_tri(I_max, J_max, conv_hexx, level)
        conv_tri_array = np.array(conv_tri)
        conv_neighbor, tri_indices, x, y, all_triangles = calculate_neighbors_2D(s, I_max, J_max, conv_hexx, level)
        if solver_type in ['forward', 'adjoint']:
            if solver_type == 'forward':
                matrix_builder = MatrixBuilderForward2DHexx(group, I_max, J_max, conv_tri, conv_neighbor, TOT, SIGS_reshaped, BC, h, level, D, chi, NUFIS)
                M, F = matrix_builder.build_forward_matrices()
            elif solver_type == 'adjoint':
                matrix_builder = MatrixBuilderAdjoint2DHexx(group, I_max, J_max, conv_tri, conv_neighbor, TOT, SIGS_reshaped, BC, h, level, D, chi, NUFIS)
                M, F = matrix_builder.build_adjoint_matrices()

            solver = SolverFactory.get_solver_power2DHexx(solver_type, group, conv_tri, M, F, h, precond, tol=1E-10)
            keff, phi_temp = solver.solve()

            PHI, PHI_reshaped, PHI_temp_reshaped = PostProcessor.postprocess_power2DHexx(phi_temp, conv_tri, group, N_hexx)
            PostProcessor.save_output_power2DHexx(output_dir, case_name, keff, PHI_reshaped, solver_type)
            for g in range(group):
                plot_triangular(PHI_temp_reshaped[g], x, y, tri_indices, g+1, cmap='viridis', varname='PHI', title=f'2D Plot of PHI{g+1} Hexx', case_name=case_name, output_dir=output_dir, solve=solver_type.upper(), process_data="magnitude")
        elif solver_type == 'noise':
            v = globals().get("v")
            Beff = globals().get("Beff")
            omega = globals().get("omega")
            l = globals().get("l")
            dTOT = globals().get("dTOT")
            dSIGS_reshaped = globals().get("dSIGS_reshaped")
            dNUFIS = globals().get("dNUFIS")
            noise_section = globals().get("noise_section")
            type_noise = globals().get("type_noise")

            # Load data from JSON file
            with open(f'{output_dir}/{case_name}_FORWARD/{case_name}_FORWARD_output.json', 'r') as json_file:
                forward_output = json.load(json_file)

            # Access keff and PHI from the loaded data
            keff = forward_output["keff"]
            PHI_all = []
            for i in range(group):
                phi_key = f"PHI{i+1}_FORWARD"
                PHI_all.append(forward_output[phi_key])

            PHI = np.zeros(max(conv_tri) * group)
            for g in range(group):
                PHI_indices = g * max(conv_tri) + (conv_tri_array - 1)
                PHI[PHI_indices] = PHI_all[g]

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

            if type_noise == 'FXV' or type_noise == 'FAV':
                if level < 2:
                    raise ValueError('Vibrating Assembly type noise only works if level at least 2')

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
            PostProcessor.save_output_fixed2DHexx(output_dir, case_name, dPHI_reshaped, solver_type)
            for g in range(group):
                plot_triangular(dPHI_temp_reshaped[g], x, y, tri_indices, g+1, cmap='viridis', varname='dPHI', title=f'2D Plot of dPHI{g+1} Hexx Magnitude', case_name=case_name, output_dir=output_dir, solve=solver_type.upper(), process_data="magnitude")
                plot_triangular(dPHI_temp_reshaped[g], x, y, tri_indices, g+1, cmap='viridis', varname='dPHI', title=f'2D Plot of dPHI{g+1} Hexx Phase', case_name=case_name, output_dir=output_dir, solve=solver_type.upper(), process_data="phase")

            S_temp_plot = dS.dot(PHI)
            S_temp_plot_reshaped = np.reshape(S_temp_plot, (group, max(conv_tri)))
            for g in range(group):
                plot_triangular(S_temp_plot_reshaped[g], x, y, tri_indices, g+1, cmap='viridis', varname='S', title=f'2D Plot of S{g+1} Hexx Magnitude', case_name=case_name, output_dir=output_dir, solve=solver_type.upper(), process_data="magnitude")
                plot_triangular(S_temp_plot_reshaped[g], x, y, tri_indices, g+1, cmap='viridis', varname='S', title=f'2D Plot of S{g+1} Hexx Phase', case_name=case_name, output_dir=output_dir, solve=solver_type.upper(), process_data="phase")

    elif geom_type =='3D rectangular':
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

        output_dir = f'OUTPUTS/{case_name}'
        Utils.create_directories(solver_type, output_dir, case_name)
        conv = convert_index_3D_rect(D, I_max, J_max, K_max)
        conv_array = np.array(conv)
        if solver_type in ['forward', 'adjoint']:
            if solver_type == 'forward':
                matrix_builder = MatrixBuilderForward3DRect(group, N, conv, TOT, SIGS_reshaped, BC, dx, dy, dz, D, chi, NUFIS)
                M, F = matrix_builder.build_forward_matrices()
            elif solver_type == 'adjoint':
                matrix_builder = MatrixBuilderAdjoint3DRect(group, N, conv, TOT, SIGS_reshaped, BC, dx, dy, dz, D, chi, NUFIS)
                M, F = matrix_builder.build_adjoint_matrices()

            solver = SolverFactory.get_solver_power3DRect(solver_type, group, N, conv, M, F, dx, dy, dz, precond, tol=1E-10)
            keff, phi_temp = solver.solve()

            PHI, PHI_reshaped, PHI_reshaped_plot = PostProcessor.postprocess_power3DRect(phi_temp, conv, group, N, I_max, J_max, K_max)
            PostProcessor.save_output_power3DRect(output_dir, case_name, keff, PHI_reshaped, solver_type)
            for g in range(group):
                image_files = []
                for k in range(K_max):
                    filename_PHI = plot_heatmap_3D(PHI_reshaped_plot[g, k, :, :], g+1, k+1, x, y, cmap='viridis', varname='PHI', title=f'2D Plot of PHI{g+1}, Z={k+1}', output_dir=output_dir, case_name=case_name, process_data='magnitude', solve=solver_type.upper())
                    image_files.append(filename_PHI)

                # Create a GIF from the saved images
                gif_filename_PHI = f'{output_dir}/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_PHI_animation_G{g+1}.gif'

                # Open images and save as GIF
                images_PHI = [Image.open(img) for img in image_files]
                images_PHI[0].save(gif_filename_PHI, save_all=True, append_images=images_PHI[1:], duration=300, loop=0)
                print(f"GIF saved as {gif_filename_PHI}")

        elif solver_type == 'noise':
            v = globals().get("v")
            Beff = globals().get("Beff")
            omega = globals().get("omega")
            l = globals().get("l")
            dTOT = globals().get("dTOT")
            dSIGS_reshaped = globals().get("dSIGS_reshaped")
            dNUFIS = globals().get("dNUFIS")

            # Load data from JSON file
            with open(f'{output_dir}/{case_name}_FORWARD/{case_name}_FORWARD_output.json', 'r') as json_file:
                forward_output = json.load(json_file)

            # Access keff and PHI from the loaded data
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
            PostProcessor.save_output_fixed3DRect(output_dir, case_name, keff, dPHI_reshaped, solver_type)
            for g in range(group):
                image_mag_files = []
                image_phase_files = []
                for k in range(K_max):
                    filename_mag = plot_heatmap_3D(dPHI_reshaped_plot[g, k, :, :], g+1, k+1, x, y, cmap='viridis', varname='dPHI', title=f'2D Plot of dPHI{g+1}, Z={k+1}, Magnitude', output_dir=output_dir, case_name=case_name, process_data='magnitude', solve=solver_type.upper())
                    filename_phase = plot_heatmap_3D(dPHI_reshaped_plot[g, k, :, :], g+1, k+1, x, y, cmap='viridis', varname='dPHI', title=f'2D Plot of dPHI{g+1}, Z={k+1}, Phase', output_dir=output_dir, case_name=case_name, process_data='phase', solve=solver_type.upper())
                    image_mag_files.append(filename_mag)
                    image_phase_files.append(filename_phase)

                # Create a GIF from the saved images
                gif_filename_mag = f'{output_dir}/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_dPHI_animation_magnitude_G{g+1}.gif'
                gif_filename_phase = f'{output_dir}/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_dPHI_animation_phase_G{g+1}.gif'

                # Open images and save as GIF
                images_mag = [Image.open(img) for img in image_mag_files]
                images_mag[0].save(gif_filename_mag, save_all=True, append_images=images_mag[1:], duration=300, loop=0)
                print(f"GIF saved as {gif_filename_mag}")

                images_phase = [Image.open(img) for img in image_phase_files]
                images_phase[0].save(gif_filename_phase, save_all=True, append_images=images_phase[1:], duration=300, loop=0)
                print(f"GIF saved as {gif_filename_phase}")

            S_temp_plot = dS.dot(PHI)
            S_plot = np.zeros(group * N, dtype=complex)
            conv_array = np.array(conv)
            non_zero_indices = np.nonzero(conv)[0]
            phi_temp_indices = conv_array[non_zero_indices] - 1

            for g in range(group):
                phi_temp_start = g * max(conv)
                S_plot[g * N + non_zero_indices] = S_temp_plot[phi_temp_start + phi_temp_indices]

            for g in range(group):
                for n in range(N):
                    if conv[n] == 0:
                        S_plot[g * N + n] = np.nan
        
            S_reshaped_plot = np.reshape(S_plot, (group, K_max, J_max, I_max))
            for g in range(group):
                image_mag_files = []
                image_phase_files = []
                for k in range(K_max):
                    filename_mag = plot_heatmap_3D(S_reshaped_plot[g, k, :, :], g+1, k+1, x, y, cmap='viridis', varname='S', title=f'2D Plot of S{g+1}, Z={k+1}, Magnitude', output_dir=output_dir, case_name=case_name, process_data='magnitude', solve=solver_type.upper())
                    filename_phase = plot_heatmap_3D(S_reshaped_plot[g, k, :, :], g+1, k+1, x, y, cmap='viridis', varname='S', title=f'2D Plot of S{g+1}, Z={k+1}, Phase', output_dir=output_dir, case_name=case_name, process_data='phase', solve=solver_type.upper())
                    image_mag_files.append(filename_mag)
                    image_phase_files.append(filename_phase)

                # Create a GIF from the saved images
                gif_filename_mag = f'{output_dir}/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_S_animation_magnitude_G{g+1}.gif'
                gif_filename_phase = f'{output_dir}/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_S_animation_phase_G{g+1}.gif'

                # Open images and save as GIF
                images_mag = [Image.open(img) for img in image_mag_files]
                images_mag[0].save(gif_filename_mag, save_all=True, append_images=images_mag[1:], duration=300, loop=0)
                print(f"GIF saved as {gif_filename_mag}")

                images_phase = [Image.open(img) for img in image_phase_files]
                images_phase[0].save(gif_filename_phase, save_all=True, append_images=images_phase[1:], duration=300, loop=0)
                print(f"GIF saved as {gif_filename_phase}")

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
        Utils.create_directories(solver_type, output_dir, case_name)
        conv_hexx = convert_3D_hexx(K_max, J_max, I_max, D)
        conv_tri, conv_hexx_ext = convert_3D_tri(K_max, J_max, I_max, conv_hexx, level)
        conv_tri_array = np.array(conv_tri)
        conv_neighbor_2D, conv_neighbor_3D, tri_indices, x, y, all_triangles = calculate_neighbors_3D(s, I_max, J_max, K_max, conv_hexx, level)
        if solver_type in ['forward', 'adjoint']:
            if solver_type == 'forward':
                matrix_builder = MatrixBuilderForward3DHexx(group, I_max, J_max, K_max, conv_tri, conv_neighbor_3D, TOT, SIGS_reshaped, BC, h, dz, level, D, chi, NUFIS)
                M, F = matrix_builder.build_forward_matrices()
            elif solver_type == 'adjoint':
                matrix_builder = MatrixBuilderAdjoint3DHexx(group, I_max, J_max, K_max, conv_tri, conv_neighbor_3D, TOT, SIGS_reshaped, BC, h, dz, level, D, chi, NUFIS)
                M, F = matrix_builder.build_adjoint_matrices()

            solver = SolverFactory.get_solver_power3DHexx(solver_type, group, conv_tri, M, F, h, dz, precond, tol=1E-10)
            keff, phi_temp = solver.solve()

            PHI, PHI_reshaped, PHI_temp_reshaped = PostProcessor.postprocess_power3DHexx(phi_temp, conv_tri, group, N_hexx, K_max, tri_indices)
            PostProcessor.save_output_power3DHexx(output_dir, case_name, keff, PHI_reshaped, solver_type)
            for g in range(group):
                image_files = []
                for k in range(K_max):
                    filename_PHI = plot_triangular_3D(PHI_temp_reshaped[g][k], x, y, k+1, tri_indices, g+1, cmap='viridis', varname='PHI', title=f'2D Plot of PHI{g+1} Hexx, Z = {k+1}', case_name=case_name, output_dir=output_dir, solve=solver_type.upper(), process_data="magnitude")
                    image_files.append(filename_PHI)

                # Create a GIF from the saved images
                gif_filename_PHI = f'{output_dir}/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_PHI_animation_G{g+1}.gif'

                # Open images and save as GIF
                images_PHI = [Image.open(img) for img in image_files]
                images_PHI[0].save(gif_filename_PHI, save_all=True, append_images=images_PHI[1:], duration=300, loop=0)
                print(f"GIF saved as {gif_filename_PHI}")
        elif solver_type == 'noise':
            v = globals().get("v")
            Beff = globals().get("Beff")
            omega = globals().get("omega")
            l = globals().get("l")
            dTOT = globals().get("dTOT")
            dSIGS_reshaped = globals().get("dSIGS_reshaped")
            dNUFIS = globals().get("dNUFIS")
            noise_section = globals().get("noise_section")
            type_noise = globals().get("type_noise")

            # Load data from JSON file
            with open(f'{output_dir}/{case_name}_FORWARD/{case_name}_FORWARD_output.json', 'r') as json_file:
                forward_output = json.load(json_file)

            # Access keff and PHI from the loaded data
            keff = forward_output["keff"]
            PHI_all = []
            for i in range(group):
                phi_key = f"PHI{i+1}_FORWARD"
                PHI_all.append(forward_output[phi_key])

            PHI = np.zeros(max(conv_tri) * group)
            for g in range(group):
                PHI_indices = g * max(conv_tri) + (conv_tri_array - 1)
                PHI[PHI_indices] = PHI_all[g]

            # Noise Input Manipulation
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

            if type_noise == 'FXV' or type_noise == 'FAV':
                if level < 2:
                    raise ValueError('Vibrating Assembly type noise only works if level at least 2')

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
            PostProcessor.save_output_fixed3DHexx(output_dir, case_name, dPHI_reshaped, solver_type)
            for g in range(group):
                image_files_mag = []
                image_files_phase = []
                for k in range(K_max):
                    filename_dPHI_mag = plot_triangular_3D(dPHI_temp_reshaped[g][k], x, y, k+1, tri_indices, g+1, cmap='viridis', varname='dPHI', title=f'2D Plot of dPHI{g+1}, Z{k+1} Hexx Magnitude', case_name=case_name, output_dir=output_dir, solve=solver_type.upper(), process_data="magnitude")
                    filename_dPHI_phase = plot_triangular_3D(dPHI_temp_reshaped[g][k], x, y, k+1, tri_indices, g+1, cmap='viridis', varname='dPHI', title=f'2D Plot of dPHI{g+1}, Z{k+1} Hexx Phase', case_name=case_name, output_dir=output_dir, solve=solver_type.upper(), process_data="phase")
                    image_files_mag.append(filename_dPHI_mag)
                    image_files_phase.append(filename_dPHI_phase)

                # Create a GIF from the saved images
                gif_filename_dPHI_mag = f'{output_dir}/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_dPHI_animation_G{g+1}_magnitude.gif'
                gif_filename_dPHI_phase = f'{output_dir}/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_dPHI_animation_G{g+1}_phase.gif'

                # Open images and save as GIF
                images_dPHI_mag = [Image.open(img) for img in image_files_mag]
                images_dPHI_phase = [Image.open(img) for img in image_files_phase]
                images_dPHI_mag[0].save(gif_filename_dPHI_mag, save_all=True, append_images=images_dPHI_mag[1:], duration=300, loop=0)
                images_dPHI_phase[0].save(gif_filename_dPHI_phase, save_all=True, append_images=images_dPHI_phase[1:], duration=300, loop=0)
                print(f"GIF saved as {gif_filename_dPHI_mag}")
                print(f"GIF saved as {gif_filename_dPHI_phase}")

            S_temp_plot = dS.dot(PHI)
            S_temp_plot_reshaped = np.reshape(S_temp_plot, (group, K_max, len(tri_indices)))
            for g in range(group):
                image_files_mag = []
                image_files_phase = []
                for k in range(K_max):
                    filename_S_mag = plot_triangular_3D(S_temp_plot_reshaped[g][k], x, y, k+1, tri_indices, g+1, cmap='viridis', varname='S', title=f'2D Plot of S{g+1}, Z{k+1} Hexx Magnitude', case_name=case_name, output_dir=output_dir, solve=solver_type.upper(), process_data="magnitude")
                    filename_S_phase = plot_triangular_3D(S_temp_plot_reshaped[g][k], x, y, k+1, tri_indices, g+1, cmap='viridis', varname='S', title=f'2D Plot of S{g+1}, Z{k+1} Hexx Phase', case_name=case_name, output_dir=output_dir, solve=solver_type.upper(), process_data="phase")
                    image_files_mag.append(filename_S_mag)
                    image_files_phase.append(filename_S_phase)

                # Create a GIF from the saved images
                gif_filename_S_mag = f'{output_dir}/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_S_animation_G{g+1}_magnitude.gif'
                gif_filename_S_phase = f'{output_dir}/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_S_animation_G{g+1}_phase.gif'

                # Open images and save as GIF
                images_S_mag = [Image.open(img) for img in image_files_mag]
                images_S_phase = [Image.open(img) for img in image_files_phase]
                images_S_mag[0].save(gif_filename_S_mag, save_all=True, append_images=images_S_mag[1:], duration=300, loop=0)
                images_S_phase[0].save(gif_filename_S_phase, save_all=True, append_images=images_S_phase[1:], duration=300, loop=0)
                print(f"GIF saved as {gif_filename_S_mag}")
                print(f"GIF saved as {gif_filename_S_phase}")

    elapsed_time = time.time() - start_time
    print(f'Time elapsed: {elapsed_time:.3e} seconds')

    # SOME INFORMATION
    info_output = f"""
    --- Simulation Summary ---
    Case Name: {case_name}
    Simulation Type: {solver_type} Simulation
    Number of groups: {group}
    Dimensions: {geom_type}
    Final keff: {keff:.6f}
    Elapsed Time: {elapsed_time:.3e} seconds
    Solver Used: {'ILU' if precond == 1 else 'LU' if precond == 2 else 'Sparse Direct Solver'}
    """

    # Save the summary to a text file
    summary_file = f'{output_dir}/{case_name}_{solver_type.upper()}/{case_name}_summary.txt'
    with open(summary_file, 'w') as file:
        file.write(info_output)

if __name__ == "__main__":
    main()
