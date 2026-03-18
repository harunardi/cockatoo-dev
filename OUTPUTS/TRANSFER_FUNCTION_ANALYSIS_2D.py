import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
from matplotlib import cm
from PIL import Image

# Prevent .pyc file generation
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True

#######################################################################################################
# 2DMG C3 V&V
case_name_C3 = 'OBJECTIVES2_TEST02_2DMG_C3_VandV'

# Load data from JSON file
with open(f'{case_name_C3}/{case_name_C3}_TRANSFER/{case_name_C3}_NOISE/{case_name_C3}_NOISE_G_0_output.json', 'r') as json_file:
    noise_output_C3 = json.load(json_file)

# Access G0_C3 numerical from the loaded data
G0_C3 = [complex(entry["real"], entry["imaginary"]) for entry in noise_output_C3["G_0"]]
G0_C3_array = np.array(G0_C3)

#######################################################################################################
# 2DMG BIBLIS
case_name_BIBLIS = 'OBJECTIVES2_TEST03_2DMG_BIBLIS_VandV'

# Load data from JSON file
with open(f'{case_name_BIBLIS}/{case_name_BIBLIS}_TRANSFER/{case_name_BIBLIS}_NOISE/{case_name_BIBLIS}_NOISE_G_0_output.json', 'r') as json_file:
    noise_output_BIBLIS = json.load(json_file)

# Access G0_BIBLIS numerical from the loaded data
G0_BIBLIS = [complex(entry["real"], entry["imaginary"]) for entry in noise_output_BIBLIS["G_0"]]
G0_BIBLIS_array = np.array(G0_BIBLIS)

#######################################################################################################
# 2DMG VVER-400
case_name_VVER = 'OBJECTIVES2_TEST05_2DTriMG_VVER400_VandV'

# Load data from JSON file
with open(f'{case_name_VVER}/{case_name_VVER}_level2_TRANSFER/{case_name_VVER}_level2_NOISE/{case_name_VVER}_level2_NOISE_G_0_output.json', 'r') as json_file:
    noise_output_VVER = json.load(json_file)

# Access G0_BIBLIS numerical from the loaded data
G0_VVER = [complex(entry["real"], entry["imaginary"]) for entry in noise_output_VVER["G_0"]]
G0_VVER_array = np.array(G0_VVER)

#######################################################################################################
# 2DTriMG HTTR
case_name_HTTR = 'OBJECTIVES2_TEST08_2DTriMG_HTTR2G_VandV'

# Load data from JSON file
with open(f'{case_name_HTTR}/{case_name_HTTR}_level2_TRANSFER/{case_name_HTTR}_level2_NOISE/{case_name_HTTR}_level2_NOISE_G_0_output.json', 'r') as json_file:
    noise_output_HTTR = json.load(json_file)

# Access G0_BIBLIS numerical from the loaded data
G0_HTTR = [complex(entry["real"], entry["imaginary"]) for entry in noise_output_HTTR["G_0"]]
G0_HTTR_array = np.array(G0_HTTR)

#######################################################################################################
freq = np.logspace(-4, 4, num=101)

# Plotting magnitude
plt.clf()  # Clear the current figure
plt.figure(figsize=(8, 6))  # Create a new figure
plt.plot(freq, abs(G0_C3_array), marker='o', label="C3 (LWR)", color='blue')
plt.plot(freq, abs(G0_BIBLIS_array), marker='s', label="BIBLIS (LWR)", color='orange')
plt.plot(freq, abs(G0_VVER_array), marker='d', label="VVER (LWR)", color='green')
plt.plot(freq, abs(G0_HTTR_array), marker='*', label="HTTR", color='red')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude of Transfer Function')
plt.title('Plot of Transfer Function (Magnitude)')
plt.legend()
plt.savefig(f'Transfer_Function_2D_Comparison_Magnitude.png')

# Plotting magnitude
plt.clf()  # Clear the current figure
plt.figure(figsize=(8, 6))  # Create a new figure
plt.plot(freq, np.degrees(np.angle(G0_C3_array)), marker='o', label="C3 (LWR)", color='blue')
plt.plot(freq, np.degrees(np.angle(G0_BIBLIS_array)), marker='s', label="BIBLIS (LWR)", color='orange')
plt.plot(freq, np.degrees(np.angle(G0_VVER_array)), marker='d', label="VVER (LWR)", color='green')
plt.plot(freq, np.degrees(np.angle(G0_HTTR_array)), marker='*', label="HTTR", color='red')
plt.xscale('log')
plt.ylim(-120, 0)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase of Transfer Function')
plt.title('Plot of Transfer Function (Phase)')
plt.legend()
plt.savefig(f'Transfer_Function_2D_Comparison_Phase.png')

#######################################################################################################
# Calculate the Mean Generation Time
