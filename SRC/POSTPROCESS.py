import numpy as np
import json

class PostProcessor:
    @staticmethod
    def save_output_power1D(output_dir, case_name, keff, phi_reshape, solver_type):
        output = {"keff": keff}
        for g in range(len(phi_reshape)):
            phi_groupname = f'PHI{g + 1}_{solver_type.upper()}'
            output[phi_groupname] = [val.real for val in phi_reshape[g]]

        with open(f'{output_dir}/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_output.json', 'w') as json_file:
            json.dump(output, json_file, indent=4)

    @staticmethod
    def save_output_fixed1D(output_dir, case_name, dPHI_reshaped, solver_type):
        output = {}
        for g in range(len(dPHI_reshaped)):
            dPHI_groupname = f'dPHI{g + 1}'
            dPHI_list = [{"real": x.real, "imaginary": x.imag} for x in dPHI_reshaped[g]]
            output[dPHI_groupname] = dPHI_list

        with open(f'{output_dir}/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_output.json', 'w') as json_file:
            json.dump(output, json_file, indent=4)

    @staticmethod
    def postprocess_power2DRect(phi_temp, conv, group, N, I_max, J_max):
        phi = np.zeros(group * N)
        conv_array = np.array(conv)
        non_zero_indices = np.nonzero(conv)[0]
        phi_temp_indices = conv_array[non_zero_indices] - 1

        for g in range(group):
            phi_temp_start = g * max(conv)
            phi[g * N + non_zero_indices] = phi_temp[
            (phi_temp_start + phi_temp_indices).astype(np.intc)
            ].real

            for n in range(N):
                if conv[n] == 0:
                    phi[g * N + n] = np.nan
        
        phi_reshaped = np.reshape(phi, (group, N))
        phi_reshaped_plot = np.reshape(phi, (group, J_max, I_max))

        return phi, phi_reshaped, phi_reshaped_plot

    @staticmethod
    def save_output_power2DRect(output_dir, case_name, keff, phi_reshape, solver_type):
        output = {"keff": keff.real}
        for g in range(len(phi_reshape)):
            phi_groupname = f'PHI{g + 1}_{solver_type.upper()}'
            output[phi_groupname] = [val.real for val in phi_reshape[g]]

        with open(f'{output_dir}/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_output.json', 'w') as json_file:
            json.dump(output, json_file, indent=4)

    @staticmethod
    def postprocess_fixed2DRect(dPHI_temp, conv, group, N, I_max, J_max):
        dPHI = np.zeros(group * N, dtype=complex)
        conv_array = np.array(conv)
        non_zero_indices = np.nonzero(conv)[0]
        phi_temp_indices = conv_array[non_zero_indices] - 1

        for g in range(group):
            dPHI_temp_start = g * max(conv)
            dPHI[g * N + non_zero_indices] = dPHI_temp[
                (dPHI_temp_start + phi_temp_indices).astype(np.intc)
            ]

            for n in range(N):
                if conv[n] == 0:
                    dPHI[g * N + n] = np.nan

        dPHI_reshaped = np.reshape(dPHI, (group, N))
        dPHI_reshaped_plot = np.reshape(dPHI, (group, J_max, I_max))

        return dPHI, dPHI_reshaped, dPHI_reshaped_plot

    @staticmethod
    def save_output_fixed2DRect(output_dir, case_name, keff, dPHI_reshape, solver_type):
        output = {}
        for g in range(len(dPHI_reshape)):
            dPHI_groupname = f'dPHI{g + 1}'
            dPHI_list = [{"real": x.real, "imaginary": x.imag} for x in dPHI_reshape[g]]
            output[dPHI_groupname] = dPHI_list

        with open(f'{output_dir}/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_output.json', 'w') as json_file:
            json.dump(output, json_file, indent=4)

    @staticmethod
    def postprocess_power2DHexx(phi_temp, conv_tri, group, N_hexx):
        phi = np.zeros(group * N_hexx)
        conv_tri_array = np.array(conv_tri)
        non_zero_indices = np.nonzero(conv_tri)[0]
        phi_temp_indices = conv_tri_array[non_zero_indices] - 1

        for g in range(group):
            phi_temp_start = g * max(conv_tri)
            phi[g * N_hexx + non_zero_indices] = phi_temp[
                (phi_temp_start + phi_temp_indices).astype(np.intc)
            ].real

            for n in range(N_hexx):
                if conv_tri[n] == 0:
                    phi[g * N_hexx + n] = np.nan
        
        phi_reshaped = np.reshape(phi, (group, N_hexx))
        phi_temp_reshaped = np.reshape(phi_temp, (group, max(conv_tri)))

        return phi, phi_reshaped, phi_temp_reshaped

    @staticmethod
    def save_output_power2DHexx(output_dir, case_name, keff, phi_reshape, solver_type):
        output = {"keff": keff.real}
        for g in range(len(phi_reshape)):
            phi_groupname = f'PHI{g + 1}_{solver_type.upper()}'
            output[phi_groupname] = [val.real for val in phi_reshape[g]]

        with open(f'{output_dir}/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_output.json', 'w') as json_file:
            json.dump(output, json_file, indent=4)

    @staticmethod
    def postprocess_fixed2DHexx(dPHI_temp, conv_tri, group, N_hexx):
        dPHI = np.zeros(group * N_hexx, dtype=complex)
        conv_tri_array = np.array(conv_tri)
        non_zero_indices = np.nonzero(conv_tri)[0]
        phi_temp_indices = conv_tri_array[non_zero_indices] - 1

        for g in range(group):
            dPHI_temp_start = g * max(conv_tri)
            dPHI[g * N_hexx + non_zero_indices] = dPHI_temp[
                (dPHI_temp_start + phi_temp_indices).astype(np.intc)
            ]

            for n in range(N_hexx):
                if conv_tri[n] == 0:
                    dPHI[g * N_hexx + n] = np.nan

        dPHI_reshaped = np.reshape(dPHI, (group, N_hexx))
        dPHI_temp_reshaped = np.reshape(dPHI_temp, (group, max(conv_tri)))

        return dPHI, dPHI_reshaped, dPHI_temp_reshaped

    @staticmethod
    def save_output_fixed2DHexx(output_dir, case_name, dPHI_reshape, solver_type):
        output = {}
        for g in range(len(dPHI_reshape)):
            dPHI_groupname = f'dPHI{g + 1}'
            dPHI_list = [{"real": x.real, "imaginary": x.imag} for x in dPHI_reshape[g]]
            output[dPHI_groupname] = dPHI_list

        with open(f'{output_dir}/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_output.json', 'w') as json_file:
            json.dump(output, json_file, indent=4)

    @staticmethod
    def postprocess_power3DRect(phi_temp, conv, group, N, I_max, J_max, K_max):
        phi = np.zeros(group * N)
        conv_array = np.array(conv)
        non_zero_indices = np.nonzero(conv)[0]
        phi_temp_indices = conv_array[non_zero_indices] - 1

        for g in range(group):
            phi_temp_start = g * max(conv)
            phi[g * N + non_zero_indices] = phi_temp[
                (phi_temp_start + phi_temp_indices).astype(np.intc)
            ].real

            for n in range(N):
                if conv[n] == 0:
                    phi[g * N + n] = np.nan
        
        phi_reshaped = np.reshape(phi, (group, N))
        phi_reshaped_plot = np.reshape(phi, (group, K_max, J_max, I_max))

        return phi, phi_reshaped, phi_reshaped_plot

    @staticmethod
    def save_output_power3DRect(output_dir, case_name, keff, phi_reshape, solver_type):
        output = {"keff": keff.real}
        for g in range(len(phi_reshape)):
            phi_groupname = f'PHI{g + 1}_{solver_type.upper()}'
            output[phi_groupname] = [val.real for val in phi_reshape[g]]

        with open(f'{output_dir}/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_output.json', 'w') as json_file:
            json.dump(output, json_file, indent=4)

    @staticmethod
    def postprocess_fixed3DRect(dPHI_temp, conv, group, N, I_max, J_max, K_max):
        dPHI = np.zeros(group * N, dtype=complex)
        conv_array = np.array(conv)
        non_zero_indices = np.nonzero(conv)[0]
        phi_temp_indices = conv_array[non_zero_indices] - 1

        for g in range(group):
            dPHI_temp_start = g * max(conv)
            dPHI[g * N + non_zero_indices] = dPHI_temp[
                (dPHI_temp_start + phi_temp_indices).astype(np.intc)
            ]

            for n in range(N):
                if conv[n] == 0:
                    dPHI[g * N + n] = np.nan

        dPHI_reshaped = np.reshape(dPHI, (group, N))
        dPHI_reshaped_plot = np.reshape(dPHI, (group, K_max, J_max, I_max))

        return dPHI, dPHI_reshaped, dPHI_reshaped_plot

    @staticmethod
    def save_output_fixed3DRect(output_dir, case_name, keff, dPHI_reshape, solver_type):
        output = {}
        for g in range(len(dPHI_reshape)):
            dPHI_groupname = f'dPHI{g + 1}'
            dPHI_list = [{"real": x.real, "imaginary": x.imag} for x in dPHI_reshape[g]]
            output[dPHI_groupname] = dPHI_list

        with open(f'{output_dir}/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_output.json', 'w') as json_file:
            json.dump(output, json_file, indent=4)

    @staticmethod
    def postprocess_power3DHexx(phi_temp, conv_tri, group, N_hexx, K_max, tri_indices):
        phi = np.zeros(group * N_hexx)
        conv_tri_array = np.array(conv_tri)
        non_zero_indices = np.nonzero(conv_tri)[0]
        phi_temp_indices = conv_tri_array[non_zero_indices] - 1

        for g in range(group):
            phi_temp_start = g * max(conv_tri)
            phi[g * N_hexx + non_zero_indices] = phi_temp[
                (phi_temp_start + phi_temp_indices).astype(np.intc)
            ].real

            for n in range(N_hexx):
                if conv_tri[n] == 0:
                    phi[g * N_hexx + n] = np.nan
        
        phi_reshaped = np.reshape(phi, (group, N_hexx))
        phi_temp_reshaped = np.reshape(phi_temp, (group, K_max, len(tri_indices)))

        return phi, phi_reshaped, phi_temp_reshaped

    @staticmethod
    def save_output_power3DHexx(output_dir, case_name, keff, phi_reshape, solver_type):
        output = {"keff": keff.real}
        for g in range(len(phi_reshape)):
            phi_groupname = f'PHI{g + 1}_{solver_type.upper()}'
            output[phi_groupname] = [val.real for val in phi_reshape[g]]

        with open(f'{output_dir}/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_output.json', 'w') as json_file:
            json.dump(output, json_file, indent=4)

    @staticmethod
    def postprocess_fixed3DHexx(dPHI_temp, conv_tri, group, N_hexx, K_max, tri_indices):
        dPHI = np.zeros(group * N_hexx, dtype=complex)
        conv_tri_array = np.array(conv_tri)
        non_zero_indices = np.nonzero(conv_tri)[0]
        phi_temp_indices = conv_tri_array[non_zero_indices] - 1

        for g in range(group):
            dPHI_temp_start = g * max(conv_tri)
            dPHI[g * N_hexx + non_zero_indices] = dPHI_temp[
                (dPHI_temp_start + phi_temp_indices).astype(np.intc)
            ]

            for n in range(N_hexx):
                if conv_tri[n] == 0:
                    dPHI[g * N_hexx + n] = np.nan

        dPHI_reshaped = np.reshape(dPHI, (group, N_hexx))
        dPHI_temp_reshaped = np.reshape(dPHI_temp, (group, K_max, len(tri_indices)))

        return dPHI, dPHI_reshaped, dPHI_temp_reshaped

    @staticmethod
    def save_output_fixed3DHexx(output_dir, case_name, dPHI_reshape, solver_type):
        output = {}
        for g in range(len(dPHI_reshape)):
            dPHI_groupname = f'dPHI{g + 1}'
            dPHI_list = [{"real": x.real, "imaginary": x.imag} for x in dPHI_reshape[g]]
            output[dPHI_groupname] = dPHI_list

        with open(f'{output_dir}/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_output.json', 'w') as json_file:
            json.dump(output, json_file, indent=4)
