import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
from matplotlib import cm
from PIL import Image
import math

# Prevent .pyc file generation
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True

#######################################################################################################
# 2DMG C3 V&V
case_name = 'OBJECTIVES45_TEST07_3DTriMG_HTTR_AVS'
group = 2
k = 3  # change to 1, 2, or 3

#######################################################################################################
# Load data from JSON file
with open(f'{case_name}/{case_name}_level1_03_INVERT/{case_name}_level1_dS_unfold_INVERT_output.json', 'r') as json_file:
    noise_output_C3 = json.load(json_file)

all_values_invert = []
index_map_invert = []   # (group_number, local_index)

for i in range(group):  # loop over groups
    invert_key = f"dS_unfold{i+1}"
    for j, entry in enumerate(noise_output_C3[invert_key]):
        real_val = entry["real"]
        imag_val = entry["imaginary"]

        if math.isnan(real_val):
            real_val = 0.0  

        value = complex(real_val, imag_val)

        all_values_invert.append(value)
        index_map_invert.append((i+1, j+1))  # store group number (1-based) and index (1-based)

# Convert to numpy array
all_values_array_invert = np.array(all_values_invert)

# Find maximum by magnitude
magnitudes = np.abs(all_values_array_invert)
sorted_indices = np.argsort(magnitudes)[::-1]  # biggest first

# Pick top-k
top_indices = sorted_indices[:k]

print(f"Top k values by magnitude (Inversion) for {case_name}:")
for rank, idx in enumerate(top_indices, start=1):
    value = all_values_array_invert[idx]
    group_num, local_index = index_map_invert[idx]
    print(f"  Top {rank}: {value}, Group: {group_num}, Index: {local_index}")

########################################################################################################
# Load data from JSON file
with open(f'{case_name}/{case_name}_level1_04_ZONE/{case_name}_level1_dS_unfold_ZONE_output.json', 'r') as json_file:
    noise_output_C3 = json.load(json_file)

all_values_zone = []
index_map_zone = []   # (group_number, local_index)

for i in range(group):  # loop over groups
    zone_key = f"dS_unfold{i+1}"
    for j, entry in enumerate(noise_output_C3[zone_key]):
        real_val = entry["real"]
        imag_val = entry["imaginary"]

        if math.isnan(real_val):
            real_val = 0.0  

        value = complex(real_val, imag_val)

        all_values_zone.append(value)
        index_map_zone.append((i+1, j+1))  # store group number (1-based) and index (1-based)

# Convert to numpy array
all_values_array_zone = np.array(all_values_zone)

# Find maximum by magnitude
magnitudes = np.abs(all_values_array_zone)
sorted_indices = np.argsort(magnitudes)[::-1]  # biggest first

# Pick top-k
top_indices = sorted_indices[:k]

print(f"Top k values by magnitude (Zoning) for {case_name}:")
for rank, idx in enumerate(top_indices, start=1):
    value = all_values_array_zone[idx]
    group_num, local_index = index_map_zone[idx]
    print(f"  Top {rank}: {value}, Group: {group_num}, Index: {local_index}")

########################################################################################################

#######################################################################################################
# 2DMG C3 V&V
case_name = 'OBJECTIVES45_TEST15_3DTriMG_HTTR_AVS2S'
group = 2
k = 3  # change to 1, 2, or 3

#######################################################################################################
# Load data from JSON file
with open(f'{case_name}/{case_name}_level1_03_INVERT/{case_name}_level1_dS_unfold_INVERT_output.json', 'r') as json_file:
    noise_output_C3 = json.load(json_file)

all_values_invert = []
index_map_invert = []   # (group_number, local_index)

for i in range(group):  # loop over groups
    invert_key = f"dS_unfold{i+1}"
    for j, entry in enumerate(noise_output_C3[invert_key]):
        real_val = entry["real"]
        imag_val = entry["imaginary"]

        if math.isnan(real_val):
            real_val = 0.0  

        value = complex(real_val, imag_val)

        all_values_invert.append(value)
        index_map_invert.append((i+1, j+1))  # store group number (1-based) and index (1-based)

# Convert to numpy array
all_values_array_invert = np.array(all_values_invert)

# Find maximum by magnitude
magnitudes = np.abs(all_values_array_invert)
sorted_indices = np.argsort(magnitudes)[::-1]  # biggest first

# Pick top-k
top_indices = sorted_indices[:k]

print(f"\nTop k values by magnitude (Inversion) for {case_name}:")
for rank, idx in enumerate(top_indices, start=1):
    value = all_values_array_invert[idx]
    group_num, local_index = index_map_invert[idx]
    print(f"  Top {rank}: {value}, Group: {group_num}, Index: {local_index}")

########################################################################################################
# Load data from JSON file
with open(f'{case_name}/{case_name}_level1_04_ZONE/{case_name}_level1_dS_unfold_ZONE_output.json', 'r') as json_file:
    noise_output_C3 = json.load(json_file)

all_values_zone = []
index_map_zone = []   # (group_number, local_index)

for i in range(group):  # loop over groups
    zone_key = f"dS_unfold{i+1}"
    for j, entry in enumerate(noise_output_C3[zone_key]):
        real_val = entry["real"]
        imag_val = entry["imaginary"]

        if math.isnan(real_val):
            real_val = 0.0  

        value = complex(real_val, imag_val)

        all_values_zone.append(value)
        index_map_zone.append((i+1, j+1))  # store group number (1-based) and index (1-based)

# Convert to numpy array
all_values_array_zone = np.array(all_values_zone)

# Find maximum by magnitude
magnitudes = np.abs(all_values_array_zone)
sorted_indices = np.argsort(magnitudes)[::-1]  # biggest first

# Pick top-k
top_indices = sorted_indices[:k]

print(f"Top k values by magnitude (Zoning) for {case_name}:")
for rank, idx in enumerate(top_indices, start=1):
    value = all_values_array_zone[idx]
    group_num, local_index = index_map_zone[idx]
    print(f"  Top {rank}: {value}, Group: {group_num}, Index: {local_index}")

########################################################################################################
# Load data from JSON file
with open(f'{case_name}/{case_name}_level1_05_SCAN/{case_name}_level1_dS_SCAN_output.json', 'r') as json_file:
    noise_output_C3 = json.load(json_file)

all_values_scan = []
index_map_scan = []   # (group_number, local_index)

for i in range(group):  # loop over groups
    scan_key = f"dS_unfold{i+1}"
    for j, entry in enumerate(noise_output_C3[scan_key]):
        real_val = entry["real"]
        imag_val = entry["imaginary"]

        if math.isnan(real_val):
            real_val = 0.0  

        value = complex(real_val, imag_val)

        all_values_scan.append(value)
        index_map_scan.append((i+1, j+1))  # store group number (1-based) and index (1-based)

# Convert to numpy array
all_values_array_scan = np.array(all_values_scan)

# Find maximum by magnitude
magnitudes = np.abs(all_values_array_scan)
sorted_indices = np.argsort(magnitudes)[::-1]  # biggest first

# Pick top-k
top_indices = sorted_indices[:k]

print(f"Top k values by magnitude (Scanning) for {case_name}:")
for rank, idx in enumerate(top_indices, start=1):
    value = all_values_array_scan[idx]
    group_num, local_index = index_map_scan[idx]
    print(f"  Top {rank}: {value}, Group: {group_num}, Index: {local_index}")

########################################################################################################

#######################################################################################################
# 2DMG C3 V&V
case_name = 'OBJECTIVES45_TEST12_3DTriMG_HTTR_AVS3S'
group = 2
k = 3  # change to 1, 2, or 3

#######################################################################################################
# Load data from JSON file
with open(f'{case_name}/{case_name}_level1_03_INVERT/{case_name}_level1_dS_unfold_INVERT_output.json', 'r') as json_file:
    noise_output_C3 = json.load(json_file)

all_values_invert = []
index_map_invert = []   # (group_number, local_index)

for i in range(group):  # loop over groups
    invert_key = f"dS_unfold{i+1}"
    for j, entry in enumerate(noise_output_C3[invert_key]):
        real_val = entry["real"]
        imag_val = entry["imaginary"]

        if math.isnan(real_val):
            real_val = 0.0  

        value = complex(real_val, imag_val)

        all_values_invert.append(value)
        index_map_invert.append((i+1, j+1))  # store group number (1-based) and index (1-based)

# Convert to numpy array
all_values_array_invert = np.array(all_values_invert)

# Find maximum by magnitude
magnitudes = np.abs(all_values_array_invert)
sorted_indices = np.argsort(magnitudes)[::-1]  # biggest first

# Pick top-k
top_indices = sorted_indices[:k]

print(f"\nTop k values by magnitude (Inversion) for {case_name}:")
for rank, idx in enumerate(top_indices, start=1):
    value = all_values_array_invert[idx]
    group_num, local_index = index_map_invert[idx]
    print(f"  Top {rank}: {value}, Group: {group_num}, Index: {local_index}")

########################################################################################################
# Load data from JSON file
with open(f'{case_name}/{case_name}_level1_04_ZONE/{case_name}_level1_dS_unfold_ZONE_output.json', 'r') as json_file:
    noise_output_C3 = json.load(json_file)

all_values_zone = []
index_map_zone = []   # (group_number, local_index)

for i in range(group):  # loop over groups
    zone_key = f"dS_unfold{i+1}"
    for j, entry in enumerate(noise_output_C3[zone_key]):
        real_val = entry["real"]
        imag_val = entry["imaginary"]

        if math.isnan(real_val):
            real_val = 0.0  

        value = complex(real_val, imag_val)

        all_values_zone.append(value)
        index_map_zone.append((i+1, j+1))  # store group number (1-based) and index (1-based)

# Convert to numpy array
all_values_array_zone = np.array(all_values_zone)

# Find maximum by magnitude
magnitudes = np.abs(all_values_array_zone)
sorted_indices = np.argsort(magnitudes)[::-1]  # biggest first

# Pick top-k
top_indices = sorted_indices[:k]

print(f"Top k values by magnitude (Zoning) for {case_name}:")
for rank, idx in enumerate(top_indices, start=1):
    value = all_values_array_zone[idx]
    group_num, local_index = index_map_zone[idx]
    print(f"  Top {rank}: {value}, Group: {group_num}, Index: {local_index}")

########################################################################################################
# Load data from JSON file
with open(f'{case_name}/{case_name}_level1_05_SCAN/{case_name}_level1_dS_SCAN_output.json', 'r') as json_file:
    noise_output_C3 = json.load(json_file)

all_values_scan = []
index_map_scan = []   # (group_number, local_index)

for i in range(group):  # loop over groups
    scan_key = f"dS_unfold{i+1}"
    for j, entry in enumerate(noise_output_C3[scan_key]):
        real_val = entry["real"]
        imag_val = entry["imaginary"]

        if math.isnan(real_val):
            real_val = 0.0  

        value = complex(real_val, imag_val)

        all_values_scan.append(value)
        index_map_scan.append((i+1, j+1))  # store group number (1-based) and index (1-based)

# Convert to numpy array
all_values_array_scan = np.array(all_values_scan)

# Find maximum by magnitude
magnitudes = np.abs(all_values_array_scan)
sorted_indices = np.argsort(magnitudes)[::-1]  # biggest first

# Pick top-k
top_indices = sorted_indices[:k]

print(f"Top k values by magnitude (Scanning) for {case_name}:")
for rank, idx in enumerate(top_indices, start=1):
    value = all_values_array_scan[idx]
    group_num, local_index = index_map_scan[idx]
    print(f"  Top {rank}: {value}, Group: {group_num}, Index: {local_index}")

########################################################################################################
