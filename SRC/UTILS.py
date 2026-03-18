import os
import numpy as np
import matplotlib.pyplot as plt
import json
import sys
from matplotlib.colors import Normalize
from matplotlib import cm
from PIL import Image
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

class Utils:
    @staticmethod
    def create_directories(solver_type, output_dir, case_name):
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f'{output_dir}/{case_name}_{solver_type.upper()}', exist_ok=True)

    @staticmethod
    def check_variables(defaults, scope):
        for var_name, default_value in defaults.items():
            if var_name not in scope:
                scope[var_name] = default_value

    @staticmethod
    def plot_1D_power(solver_type, data, x, g, output_dir=None, varname=None, case_name=None, title=None):
        plt.clf()

        plt.figure()
        plt.plot(x, data, 'g-', label=f'Group {g+1} - Magnitude - {varname}_{solver_type.upper()}')
        plt.legend()
        plt.ylabel('Normalized amplitude')
        plt.title(f'Magnitude G{g+1}')
        plt.xlabel('Distance from core centre [cm]')
        plt.grid()
        filename = f'{output_dir}/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_{varname}_G{g+1}'
        plt.savefig(filename)
        plt.close()

        return filename

    @staticmethod
    def plot_1D_fixed(solver_type, data, x, g, output_dir=None, varname=None, case_name=None, title=None):
        plt.clf()

        plt.figure()
        plt.plot(x, np.abs(data)/np.max(np.abs(data)), 'g-', label=f'Group {g+1} - {varname}_{solver_type.upper()}')
        plt.legend()
        plt.ylabel('Normalized amplitude')
        plt.title(f'Magnitude G{g+1}')
        plt.xlabel('Distance from core centre [cm]')
        plt.grid()
        filename = f'{output_dir}/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_magnitude_{varname}_G{g+1}'
        plt.savefig(filename)
        plt.close()

        plt.figure()
        plt.plot(x, np.degrees(np.angle(data)), 'g-', label=f'Group {g+1} - Phase - {varname}_{solver_type.upper()}')
        plt.legend()
        plt.ylabel('Normalized amplitude')
        plt.title(f'Magnitude G{g+1}')
        plt.xlabel('Distance from core centre [cm]')
        plt.grid()
        filename = f'{output_dir}/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_phase_{varname}_G{g+1}'
        plt.savefig(filename)
        plt.close()

        return filename

    @staticmethod
    def plot_2D_rect_power(solver_type, data, x, y, g, cmap='viridis', output_dir=None, varname=None, case_name=None, title=None):
        plt.clf()

        extent = [x.min(), x.max(), y.min(), y.max()]
        plt.imshow(data, cmap=cmap, interpolation='nearest', extent=extent, origin='lower')

        plt.colorbar()  # Add color bar to show scale
        if title:
            plt.title(title)
        plt.xlabel('X (cm)')
        plt.ylabel('Y (cm)')

        x_ticks = np.linspace(x.min(), x.max(), num=10)
        y_ticks = np.linspace(y.min(), y.max(), num=10)
        plt.xticks(x_ticks, labels=[f'{val:.1f}' for val in x_ticks])
        plt.yticks(y_ticks, labels=[f'{val:.1f}' for val in y_ticks])

        filename = f'{output_dir}/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_{varname}_G{g}.png'
        plt.savefig(filename)
        plt.close()

        return filename

    @staticmethod
    def plot_2D_rect_fixed(solver_type, data, x, y, g, cmap='viridis', output_dir=None, varname=None, case_name=None, title=None, process_data=None):
        plt.clf()
        if process_data == 'magnitude':
            data = np.abs(data)  # Compute magnitude
        elif process_data == 'phase':
            data_rad = np.angle(data)  # Compute phase
            data = np.degrees(data_rad)  # Compute phase

        extent = [x.min(), x.max(), y.min(), y.max()]
        plt.imshow(data, cmap=cmap, interpolation='nearest', extent=extent, origin='lower')

        if process_data == 'magnitude':
            plt.colorbar(label=f'{varname}{g}_mag')  # Add color bar to show scale
        elif process_data == 'phase':
            plt.colorbar(label=f'{varname}{g}_deg')  # Add color bar to show scale

        if title:
            plt.title(title)
        plt.xlabel('X (cm)')
        plt.ylabel('Y (cm)')

        x_ticks = np.linspace(x.min(), x.max(), num=10)
        y_ticks = np.linspace(y.min(), y.max(), num=10)
        plt.xticks(x_ticks, labels=[f'{val:.1f}' for val in x_ticks])
        plt.yticks(y_ticks, labels=[f'{val:.1f}' for val in y_ticks])

        filename = f'{output_dir}/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_{varname}_{process_data}_G{g}.png'
        plt.savefig(filename)
        plt.close()

        return filename

    @staticmethod
    def plot_2D_rect_fixed_general(solver_type, data, x, y, g, cmap='viridis', output=None, varname=None, case_name=None, title=None, process_data=None):
        plt.clf()
        if process_data == 'magnitude':
            data = np.abs(data)  # Compute magnitude
        elif process_data == 'phase':
            data_rad = np.angle(data)  # Compute phase
            data = np.degrees(data_rad)  # Compute phase

        extent = [x.min(), x.max(), y.min(), y.max()]
        plt.imshow(data, cmap=cmap, interpolation='nearest', extent=extent, origin='lower')

        if process_data == 'magnitude':
            plt.colorbar(label=f'{varname}{g}_mag')  # Add color bar to show scale
        elif process_data == 'phase':
            plt.colorbar(label=f'{varname}{g}_deg')  # Add color bar to show scale

        if title:
            plt.title(title)
        plt.xlabel('X (cm)')
        plt.ylabel('Y (cm)')

        x_ticks = np.linspace(x.min(), x.max(), num=10)
        y_ticks = np.linspace(y.min(), y.max(), num=10)
        plt.xticks(x_ticks, labels=[f'{val:.1f}' for val in x_ticks])
        plt.yticks(y_ticks, labels=[f'{val:.1f}' for val in y_ticks])

        filename = f'{output}_{varname}_{process_data}_G{g}.png'
        plt.savefig(filename)
        plt.close()

        return filename

    @staticmethod
    def plot_2D_rect_fixed_categorical(solver_type, data, x, y, g,
                                   output=None, varname=None, case_name=None, title=None):
        plt.clf()

        # Define categories and corresponding colors
        categories = [0, 0.5, 1, 1.5]
        colors = ['lightgray', 'blue', 'red', 'purple']
        labels = ['Empty', 'Detector', 'Source', 'Detector + Source']

        # Create a discrete colormap
        cmap = mcolors.ListedColormap(colors)
        bounds = [-0.25, 0.25, 0.75, 1.25, 1.75]  # boundaries between categories
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        extent = [x.min(), x.max(), y.min(), y.max()]
        plt.imshow(data, cmap=cmap, norm=norm, interpolation='nearest', extent=extent, origin='lower')

        # Build legend manually (instead of colorbar)
        legend_elements = [Patch(facecolor=colors[i], edgecolor='k', label=labels[i]) for i in range(len(categories))]
        plt.legend(handles=legend_elements,
               loc='upper center',          # center it
               bbox_to_anchor=(0.5, -0.12), # shift below the plot
               ncol=len(categories),        # put all items in one row
               frameon=False)               # remove box around legend

        if title:
            plt.title(title)
        plt.xlabel('X (cm)')
        plt.ylabel('Y (cm)')

        x_ticks = np.linspace(x.min(), x.max(), num=10)
        y_ticks = np.linspace(y.min(), y.max(), num=10)
        plt.xticks(x_ticks, labels=[f'{val:.1f}' for val in x_ticks])
        plt.yticks(y_ticks, labels=[f'{val:.1f}' for val in y_ticks])

        filename = f'{output}_{varname}_categorical_G{g}.png'
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.05)
        plt.close()

        return filename

#    @staticmethod
#    def generate_dPHI_gif(solver_type, dPHI, f, group, N, I_max, J_max, output_dir, case_name, max_time=2, num_timesteps=201):
#        # Define time steps
#        time_steps = np.linspace(0, max_time, num_timesteps)
#
#        # Create separate lists for filenames for each group
#        group_filenames = [[] for _ in range(group)]
#
#        z_limit_g = []
#        z_magnitude = []
#        dPHI_array = np.array(dPHI)
#        dPHI_array = np.nan_to_num(dPHI_array, nan=0)
#        for g in range(group):
#            for n in range(N):
#                z_magnitude.append(np.abs(dPHI_array[g*N + n]))
#        z_magnitude_array = np.array(z_magnitude).reshape(group, N)
#        for g in range(group):
#            z_limit_g.append(max(z_magnitude_array[g]) * 1.1)
#
#        for i, t in enumerate(time_steps):
#            print(f'Plotting dPHI_time for timestep t = {t} s')
#            dPHI_time = dPHI.copy()
#
#            for g in range(group):
#                for n in range(N):
#                    magnitude = np.abs(dPHI[g*N + n])
#                    phase = np.angle(dPHI[g*N + n])
#
#                    dPHI_time[g*N + n] = magnitude * np.cos(2 * np.pi * f * t + phase)
#
#            dPHI_time_array = np.array(dPHI_time)
#            dPHI_time_reshaped = dPHI_time_array.reshape(group, I_max, J_max)
#
#            for g in range(group):
#                z_limit = z_limit_g[g] #np.nanmax(np.abs(dPHI_time_reshaped[g])) * 1.1
#                fig = plt.figure(figsize=(10, 7))
#                ax = fig.add_subplot(111, projection='3d')
#
#                # Create X, Y coordinates for bars
#                x, y = np.meshgrid(np.arange(I_max), np.arange(J_max), indexing='ij')
#                x = x.flatten()
#                y = y.flatten()
#
#                # Set bar positions and heights based on the magnitude of dPHI values
#                z = np.zeros_like(x)
#                dx = dy = 0.8  # bar width
#                dz = np.abs(dPHI_time_reshaped[g, :, :]).flatten()  # height as magnitude of complex number
#
#                # Normalize the colors based on bar height for color mapping
#                norm = Normalize(vmin=0, vmax=z_limit)
#                colors = cm.viridis(norm(dz))  # Apply colormap (viridis can be changed to any colormap)
#
#                # Plot bars with color
#                ax.bar3d(x, y, z, dx, dy, dz, color=colors, shade=True)
#
#                # Add color bar
#                mappable = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
#                mappable.set_array(dz)
#                cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10)
#                cbar.set_label('|dPHI|')
#
#                # Setting a consistent limit for the z-axis
#                ax.set_zlim(0, z_limit)
#
#                # Labeling
#                ax.set_title(f'dPHI in Time Domain for Group {g + 1}\nt = {t:.2f} s')
#                ax.set_xlabel('X index')
#                ax.set_ylabel('Y index')
#                ax.set_zlabel('|dPHI|')
#                plt.tight_layout()
#
#                # Save frame for each group
#                frame_filename = f'{output_dir}/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_time_G{g+1}_t{i:03}.png'
#                plt.savefig(frame_filename)
#                group_filenames[g].append(frame_filename)
#                plt.close(fig)  # Close figure to save memory
#
#        # Generate a GIF for each group
#        for g, filenames in enumerate(group_filenames):
#            print(f'Making GIF for dPHI_time group {g+1}')
#            frames = [Image.open(filename) for filename in filenames]
#            frames[0].save(
#                f'{output_dir}/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_time_G{g+1}_animation.gif',
#                save_all=True,
#                append_images=frames[1:],  # Append remaining frames
#                duration=300,               # Duration for each frame in ms
#                loop=0                      # Loop forever
#            )
#
#            # Optional: Cleanup (delete individual frames if not needed anymore)
#            for filename in filenames:
#                os.remove(filename)
#
#