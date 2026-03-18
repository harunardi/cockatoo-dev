import json
from collections import defaultdict
import os
import sys

# Prevent .pyc file generation
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True

import json
from pathlib import Path
import matplotlib.pyplot as plt

group = 2
for g in range(group):
# -------------------------------------------------------
# 1. Find all flux_data.json files recursively
# -------------------------------------------------------
    root_dir = Path(".")   # or specify absolute path
    json_files = list(root_dir.rglob(f"flux{g+1}_magnitude.json"))

    print(f"Found {len(json_files)} flux_data.json files")

# -------------------------------------------------------
# 2. Plot all magnitude data files
# -------------------------------------------------------
    plt.figure(figsize=(10, 6))

    for json_path in json_files:
        with open(json_path, "r") as f:
            data = json.load(f)

        unique_distances = data["unique_distances"]
        flux1_values = data["flux1_values"]

        folder = json_path.parent.name
        label_prefix = next((part for part in folder.split("_") if part.startswith("EXPAND")), folder)
        plt.plot(unique_distances, flux1_values, label=f"{label_prefix} – PK")

    plt.xlabel('Distance to Core Center (cm)')
    plt.ylabel(f"dPHI{g}")
    plt.title(fr'$\delta \phi_{{{g+1}}}^{{\text{{pk}}}}$ at Centerline (y=0)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"combined_dPHI{g+1}_PK_magnitude_comparison.png")

    plt.figure(figsize=(10, 6))

    for json_path in json_files:
        with open(json_path, "r") as f:
            data = json.load(f)

        unique_distances = data["unique_distances"]
        flux2_values = data["flux2_values"]

        folder = json_path.parent.name
        label_prefix = next((part for part in folder.split("_") if part.startswith("EXPAND")), folder)
        plt.plot(unique_distances, flux2_values, label=f"{label_prefix} – SPATIAL")

    plt.xlabel('Distance to Core Center (cm)')
    plt.ylabel(f"dPHI{g}")
    plt.title(fr'$\delta \phi_{{{g+1}}}^{{\text{{spatial}}}}$ at Centerline (y=0)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"combined_dPHI{g+1}_SPATIAL_magnitude_comparison.png")

# -------------------------------------------------------
# 3. Find all flux_data.json files recursively
# -------------------------------------------------------
    root_dir = Path(".")   # or specify absolute path
    json_files = list(root_dir.rglob(f"flux{g+1}_phase.json"))

    print(f"Found {len(json_files)} flux_data.json files")

# -------------------------------------------------------
# 2. Plot all magnitude data files
# -------------------------------------------------------
    plt.figure(figsize=(10, 6))

    for json_path in json_files:
        with open(json_path, "r") as f:
            data = json.load(f)

        unique_distances = data["unique_distances"]
        flux1_values = data["flux1_values"]

        folder = json_path.parent.name
        label_prefix = next((part for part in folder.split("_") if part.startswith("EXPAND")), folder)
        plt.plot(unique_distances, flux1_values, label=f"{label_prefix} – PK")

    plt.xlabel('Distance to Core Center (cm)')
    plt.ylabel(f"dPHI{g}")
    plt.title(fr'$\delta \phi_{{{g+1}}}^{{\text{{pk}}}}$ at Centerline (y=0)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"combined_dPHI{g+1}_PK_phase_comparison.png")

    plt.figure(figsize=(10, 6))

    for json_path in json_files:
        with open(json_path, "r") as f:
            data = json.load(f)

        unique_distances = data["unique_distances"]
        flux2_values = data["flux2_values"]

        folder = json_path.parent.name
        label_prefix = next((part for part in folder.split("_") if part.startswith("EXPAND")), folder)
        plt.plot(unique_distances, flux2_values, label=f"{label_prefix} – SPATIAL")

    plt.xlabel('Distance to Core Center (cm)')
    plt.ylabel(f"dPHI{g}")
    plt.title(fr'$\delta \phi_{{{g+1}}}^{{\text{{spatial}}}}$ at Centerline (y=0)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"combined_dPHI{g+1}_SPATIAL_phase_comparison.png")
