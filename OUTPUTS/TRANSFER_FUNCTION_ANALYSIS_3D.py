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
# 3DMG PWR
case_name_PWR = 'OBJECTIVES2_TEST06_3DMG_CSTest09_VandV_new'

# Load data from JSON file
with open(f'{case_name_PWR}/{case_name_PWR}_TRANSFER/{case_name_PWR}_NOISE/{case_name_PWR}_NOISE_G_0_output.json', 'r') as json_file:
    noise_output_PWR = json.load(json_file)

# Access G0_PWR numerical from the loaded data
G0_PWR = [complex(entry["real"], entry["imaginary"]) for entry in noise_output_PWR["G_0"]]
G0_PWR_array = np.array(G0_PWR)

#######################################################################################################
# 3DMG Langenbuch
case_name_Langenbuch = 'OBJECTIVES2_TEST10_3DMG_Langenbuch'

# Load data from JSON file
with open(f'{case_name_Langenbuch}/{case_name_Langenbuch}_TRANSFER/{case_name_Langenbuch}_NOISE/{case_name_Langenbuch}_NOISE_G_0_output.json', 'r') as json_file:
    noise_output_Langenbuch = json.load(json_file)

# Access G0_Langenbuch numerical from the loaded data
G0_Langenbuch = [complex(entry["real"], entry["imaginary"]) for entry in noise_output_Langenbuch["G_0"]]
G0_Langenbuch_array = np.array(G0_Langenbuch)

#######################################################################################################
# 3DMG VVER-400
case_name_VVER = 'OBJECTIVES2_TEST07_3DTriMG_VVER400_VandV'

# Load data from JSON file
with open(f'{case_name_VVER}/{case_name_VVER}_level1_TRANSFER/{case_name_VVER}_level1_NOISE/{case_name_VVER}_level1_NOISE_G_0_output.json', 'r') as json_file:
    noise_output_VVER = json.load(json_file)

# Access G0_BIBLIS numerical from the loaded data
G0_VVER = [complex(entry["real"], entry["imaginary"]) for entry in noise_output_VVER["G_0"]]
G0_VVER_array = np.array(G0_VVER)

#######################################################################################################
# 3DTriMG HTTR
case_name_HTTR = 'OBJECTIVES2_TEST09_3DTriMG_HTTR'

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
plt.plot(freq, abs(G0_PWR_array), marker='o', label="PWR (LWR)", color='blue')
plt.plot(freq, abs(G0_Langenbuch_array), marker='s', label="Langenbuch (LWR)", color='orange')
plt.plot(freq, abs(G0_VVER_array), marker='d', label="VVER (LWR)", color='green')
plt.plot(freq, abs(G0_HTTR_array), marker='*', label="HTTR", color='red')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude of Transfer Function')
plt.title('Plot of Transfer Function (Magnitude)')
plt.legend()
plt.savefig(f'Transfer_Function_3D_Comparison_Magnitude.png')

# Plotting magnitude
plt.clf()  # Clear the current figure
plt.figure(figsize=(8, 6))  # Create a new figure
plt.plot(freq, np.degrees(np.angle(G0_PWR_array)), marker='o', label="PWR (LWR)", color='blue')
plt.plot(freq, np.degrees(np.angle(G0_Langenbuch_array)), marker='s', label="Langenbuch (LWR)", color='orange')
plt.plot(freq, np.degrees(np.angle(G0_VVER_array)), marker='d', label="VVER (LWR)", color='green')
plt.plot(freq, np.degrees(np.angle(G0_HTTR_array)), marker='*', label="HTTR", color='red')
plt.xscale('log')
plt.ylim(-120, 0)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase of Transfer Function')
plt.title('Plot of Transfer Function (Phase)')
plt.legend()
plt.savefig(f'Transfer_Function_3D_Comparison_Phase.png')
