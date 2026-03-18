import os
import re
import numpy as np

#================================================================================
pattern = [
    [0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0], 
    [0, 0, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 0, 0], 
    [0, 3, 3, 4, 4, 8, 1, 1, 1, 1, 1, 8, 4, 4, 3, 3, 0], 
    [0, 3, 4, 4, 5, 1, 7, 1, 7, 1, 7, 1, 5, 4, 4, 3, 0], 
    [3, 3, 4, 5, 2, 8, 1, 1, 1, 1, 1, 8, 2, 5, 4, 3, 3], 
    [3, 4, 8, 1, 8, 2, 8, 2, 6, 2, 8, 2, 8, 1, 8, 4, 3], 
    [3, 4, 1, 7, 1, 8, 1, 8, 2, 8, 1, 8, 1, 7, 1, 4, 3], 
    [3, 4, 1, 1, 1, 2, 8, 1, 8, 1, 8, 2, 1, 1, 1, 4, 3], 
    [3, 4, 1, 7, 1, 6, 2, 8, 1, 8, 2, 6, 1, 7, 1, 4, 3], 
    [3, 4, 1, 1, 1, 2, 8, 1, 8, 1, 8, 2, 1, 1, 1, 4, 3], 
    [3, 4, 1, 7, 1, 8, 1, 8, 2, 8, 1, 8, 1, 7, 1, 4, 3], 
    [3, 4, 8, 1, 8, 2, 8, 2, 6, 2, 8, 2, 8, 1, 8, 4, 3], 
    [3, 3, 4, 5, 2, 8, 1, 1, 1, 1, 1, 8, 2, 5, 4, 3, 3], 
    [0, 3, 4, 4, 5, 1, 7, 1, 7, 1, 7, 1, 5, 4, 4, 3, 0], 
    [0, 3, 3, 4, 4, 8, 1, 1, 1, 1, 1, 8, 4, 4, 3, 3, 0], 
    [0, 0, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 0, 0], 
    [0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0]
    ]

expandXS = 3
# Note: For Forward and Adjoint, expandXS = 40; for Noise
# Use ILU
multiplicator = np.ones((expandXS ,expandXS))
pattern_full_temp = np.kron(pattern, multiplicator)
pattern_full = pattern_full_temp.tolist()
J_max = len(pattern_full)
I_max = len(pattern_full[0])

D_values = [
    # Mat 1,      Mat 2,       Mat 3,       Mat 4,       Mat 5,       Mat 6,       Mat 7,       Mat 8,    
    [1.43600E+00, 1.43660E+00, 1.32000E+00, 1.43890E+00, 1.43810E+00, 1.43850E+00, 1.43890E+00, 1.43930E+00], # G1
    [0.36350E+00, 0.36360E+00, 0.27720E+00, 0.36380E+00, 0.36650E+00, 0.36650E+00, 0.36790E+00, 0.36800E+00], # G2
]

ABS_values = [
    # Mat 1,    Mat 2,     Mat 3,     Mat 4,     Mat 5,     Mat 6,     Mat 7,     Mat 8,      
    [0.0095042, 0.0096785, 0.0026562, 0.0103630, 0.0100030, 0.0101320, 0.0101650, 0.0102940], # G1
    [0.0750580, 0.0784360, 0.0715960, 0.0914080, 0.0848280, 0.0873140, 0.0880240, 0.0905100], # G2
]

NUFIS_values = [
    # Mat 1,    Mat 2,     Mat 3,     Mat 4,     Mat 5,     Mat 6,     Mat 7,     Mat 8,      
    [0.0058708, 0.0061908, 0.0000000, 0.0074527, 0.0061908, 0.0064285, 0.0061908, 0.0064285], # G1
    [0.0960670, 0.1035800, 0.0000000, 0.1323600, 0.1035800, 0.1091100, 0.1035800, 0.1091100], # G2
]

CHI_values = [
    # Mat 1,      Mat 2,       Mat 3,       Mat 4,       Mat 5,       Mat 6,       Mat 7,       Mat 8,      
    [1.00000E+00, 1.00000E+00, 1.00000E+00, 1.00000E+00, 1.00000E+00, 1.00000E+00, 1.00000E+00, 1.00000E+00], # G1
    [0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00], # G2
]

                # Mat 1,      Mat 2,       Mat 3,       Mat 4,       Mat 5,       Mat 6,       Mat 7,       Mat 8,  
SIGS11_values = [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
SIGS21_values = [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]

SIGS12_values = [0.017754, 0.017621, 0.023106, 0.017101, 0.017290, 0.017192, 0.017125, 0.017027]
SIGS22_values = [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]

group = len(ABS_values)
TOT = [[[0.0 for _ in range(I_max)] for _ in range(J_max)] for _ in range(group)]
ABS = [[[0.0 for _ in range(I_max)] for _ in range(J_max)] for _ in range(group)]
NUFIS = [[[0.0 for _ in range(I_max)] for _ in range(J_max)] for _ in range(group)]
CHI = [[[0.0 for _ in range(I_max)] for _ in range(J_max)] for _ in range(group)]
D = [[[0.0 for _ in range(I_max)] for _ in range(J_max)] for _ in range(group)]
SIGS11 = [[0.0 for _ in range(I_max)] for _ in range(J_max)]
SIGS12 = [[0.0 for _ in range(I_max)] for _ in range(J_max)]
SIGS21 = [[0.0 for _ in range(I_max)] for _ in range(J_max)]
SIGS22 = [[0.0 for _ in range(I_max)] for _ in range(J_max)]

for g in range(group):
    for j in range(J_max):
        for i in range(I_max):
            value = pattern_full[j][i]  # Assuming pattern_full is properly defined
            if value == 1:
                ABS[g][j][i] = ABS_values[g][0]
                NUFIS[g][j][i] = NUFIS_values[g][0]
                CHI[g][j][i] = CHI_values[g][0]
                D[g][j][i] = D_values[g][0]
            elif value == 2:
                ABS[g][j][i] = ABS_values[g][1]
                NUFIS[g][j][i] = NUFIS_values[g][1]
                CHI[g][j][i] = CHI_values[g][1]
                D[g][j][i] = D_values[g][1]
            elif value == 3:
                ABS[g][j][i] = ABS_values[g][2]
                NUFIS[g][j][i] = NUFIS_values[g][2]
                CHI[g][j][i] = CHI_values[g][2]
                D[g][j][i] = D_values[g][2]
            elif value == 4:
                ABS[g][j][i] = ABS_values[g][3]
                NUFIS[g][j][i] = NUFIS_values[g][3]
                CHI[g][j][i] = CHI_values[g][3]
                D[g][j][i] = D_values[g][3]
            elif value == 5:
                ABS[g][j][i] = ABS_values[g][4]
                NUFIS[g][j][i] = NUFIS_values[g][4]
                CHI[g][j][i] = CHI_values[g][4]
                D[g][j][i] = D_values[g][4]
            elif value == 6:
                ABS[g][j][i] = ABS_values[g][5]
                NUFIS[g][j][i] = NUFIS_values[g][5]
                CHI[g][j][i] = CHI_values[g][5]
                D[g][j][i] = D_values[g][5]
            elif value == 7:
                ABS[g][j][i] = ABS_values[g][6]
                NUFIS[g][j][i] = NUFIS_values[g][6]
                CHI[g][j][i] = CHI_values[g][6]
                D[g][j][i] = D_values[g][6]
            elif value == 8:
                ABS[g][j][i] = ABS_values[g][7]
                NUFIS[g][j][i] = NUFIS_values[g][7]
                CHI[g][j][i] = CHI_values[g][7]
                D[g][j][i] = D_values[g][7]

for j in range(J_max):
    for i in range(I_max):
        value = pattern_full[j][i]  # Assuming pattern_full is properly defined
        if value == 1:
            SIGS11[j][i] = SIGS11_values[0]
            SIGS12[j][i] = SIGS12_values[0]
            SIGS21[j][i] = SIGS21_values[0]
            SIGS22[j][i] = SIGS22_values[0]
        elif value == 2:
            SIGS11[j][i] = SIGS11_values[1]
            SIGS12[j][i] = SIGS12_values[1]
            SIGS21[j][i] = SIGS21_values[1]
            SIGS22[j][i] = SIGS22_values[1]
        elif value == 3:
            SIGS11[j][i] = SIGS11_values[2]
            SIGS12[j][i] = SIGS12_values[2]
            SIGS21[j][i] = SIGS21_values[2]
            SIGS22[j][i] = SIGS22_values[2]
        elif value == 4:
            SIGS11[j][i] = SIGS11_values[3]
            SIGS12[j][i] = SIGS12_values[3]
            SIGS21[j][i] = SIGS21_values[3]
            SIGS22[j][i] = SIGS22_values[3]
        elif value == 5:
            SIGS11[j][i] = SIGS11_values[4]
            SIGS12[j][i] = SIGS12_values[4]
            SIGS21[j][i] = SIGS21_values[4]
            SIGS22[j][i] = SIGS22_values[4]
        elif value == 6:
            SIGS11[j][i] = SIGS11_values[5]
            SIGS12[j][i] = SIGS12_values[5]
            SIGS21[j][i] = SIGS21_values[5]
            SIGS22[j][i] = SIGS22_values[5]
        elif value == 7:
            SIGS11[j][i] = SIGS11_values[6]
            SIGS12[j][i] = SIGS12_values[6]
            SIGS21[j][i] = SIGS21_values[6]
            SIGS22[j][i] = SIGS22_values[6]
        elif value == 8:
            SIGS11[j][i] = SIGS11_values[7]
            SIGS12[j][i] = SIGS12_values[7]
            SIGS21[j][i] = SIGS21_values[7]
            SIGS22[j][i] = SIGS22_values[7]

# Perform element-wise addition
for j in range(J_max):  # Iterate over columns
    for i in range(I_max):  # Iterate over rows
        TOT[0][j][i] = ABS[0][j][i] + SIGS12[j][i]
        TOT[1][j][i] = ABS[1][j][i] + SIGS21[j][i]

# Reshaping
N = I_max*J_max
TOT_reshaped = [[None] * N for _ in range(group)]
NUFIS_reshaped = [[None] * N for _ in range(group)]
chi_reshaped = [[None] * N for _ in range(group)]
D_reshaped = [[None] * N for _ in range(group)]
SIGS11_reshaped = [0.0 for _ in range(N)]
SIGS12_reshaped = [0.0 for _ in range(N)]
SIGS21_reshaped = [0.0 for _ in range(N)]
SIGS22_reshaped = [0.0 for _ in range(N)]

for g in range(group):
    for j in range(J_max):  
        for i in range(I_max):
            m = j * I_max + i
            TOT_reshaped[g][m] = TOT[g][j][i]
            NUFIS_reshaped[g][m] = NUFIS[g][j][i]
            chi_reshaped[g][m] = CHI[g][j][i]
            SIGS11_reshaped[m] = SIGS11[j][i]
            SIGS12_reshaped[m] = SIGS12[j][i]
            SIGS21_reshaped[m] = SIGS21[j][i]
            SIGS22_reshaped[m] = SIGS22[j][i]

TOT = TOT_reshaped
NUFIS = NUFIS_reshaped
chi = chi_reshaped
SIGS_reshaped = [[SIGS11_reshaped, SIGS21_reshaped], 
                 [SIGS12_reshaped, SIGS22_reshaped],]

#================================================================================
v1 = 1.25e+7
v2 = 2.5e+5
Beff = 0.0065
l = 0.0784130

## INITIALIZATION
case_name = "OBJECTIVES45_TEST02_2DMG_BIBLIS3_FAV"
geom_type = '2D rectangular'
precond = 2
I_max = len(D[0][0]) # N row
J_max = len(D[0]) # N column
N = I_max*J_max
dy = 23.1226/expandXS
dx = 23.1226/expandXS
x = np.arange(0, I_max*dx, dx)
y = np.arange(0, J_max*dy, dy)
group = 2
f = 1.000000e+00 
omega = 2 * np.pi * f

# BC
BC = [3, 3, 3, 3] # N, S, E, W

# CROSS SECTION DEFINITIONS
dTOT1 = [[0] * I_max for _ in range(J_max)]
dTOT2 = [[0] * I_max for _ in range(J_max)]
dNUFIS1 = [[0] * I_max for _ in range(J_max)]
dNUFIS2 = [[0] * I_max for _ in range(J_max)]
dSIGS12 = [[0] * I_max for _ in range(J_max)]
dSIGS21 = [[0] * I_max for _ in range(J_max)]
dSIGS11 = [[0] * I_max for _ in range(J_max)]
dSIGS22 = [[0] * I_max for _ in range(J_max)]

# Central noise source
type_noise = 'FAV'
bottom_row = 6 * expandXS
top_row = 7 * expandXS
left_col = 6 * expandXS
right_col = 7 * expandXS
for j in range(bottom_row, top_row):
    dTOT1[j][left_col - 1] = (-1j * np.pi / 2) * (TOT[0][j * I_max + left_col - 1] - TOT[0][j * I_max + left_col])  # left boundary
    dTOT1[j][left_col] = (-1j * np.pi / 2) * (TOT[0][j * I_max + left_col - 1] - TOT[0][j * I_max + left_col])  # left boundary
    dTOT1[j][right_col - 1] = (-1j * np.pi / 2) * (TOT[0][j * I_max + right_col - 1] - TOT[0][j * I_max + right_col])  # right boundary
    dTOT1[j][right_col] = (-1j * np.pi / 2) * (TOT[0][j * I_max + right_col - 1] - TOT[0][j * I_max + right_col])  # right boundary
    dTOT2[j][left_col - 1] = (-1j * np.pi / 2) * (TOT[1][j * I_max + left_col - 1] - TOT[1][j * I_max + left_col])  # left boundary
    dTOT2[j][left_col] = (-1j * np.pi / 2) * (TOT[1][j * I_max + left_col - 1] - TOT[1][j * I_max + left_col])  # left boundary
    dTOT2[j][right_col - 1] = (-1j * np.pi / 2) * (TOT[1][j * I_max + right_col - 1] - TOT[1][j * I_max + right_col])  # right boundary
    dTOT2[j][right_col] = (-1j * np.pi / 2) * (TOT[1][j * I_max + right_col - 1] - TOT[1][j * I_max + right_col])  # right boundary
    dNUFIS1[j][left_col - 1] = (-1j * np.pi / 2) * (NUFIS[0][j * I_max + left_col - 1] - NUFIS[0][j * I_max + left_col])  # left boundary
    dNUFIS1[j][left_col] = (-1j * np.pi / 2) * (NUFIS[0][j * I_max + left_col - 1] - NUFIS[0][j * I_max + left_col])  # left boundary
    dNUFIS1[j][right_col - 1] = (-1j * np.pi / 2) * (NUFIS[0][j * I_max + right_col - 1] - NUFIS[0][j * I_max + right_col])  # right boundary
    dNUFIS1[j][right_col] = (-1j * np.pi / 2) * (NUFIS[0][j * I_max + right_col - 1] - NUFIS[0][j * I_max + right_col])  # right boundary

#    print(NUFIS[0][j * I_max + left_col - 1], NUFIS[0][j * I_max + left_col], dNUFIS1[j][left_col], NUFIS[0][j * I_max + right_col - 1], NUFIS[0][j * I_max + right_col], dNUFIS1[j][right_col])

    dNUFIS2[j][left_col - 1] = (-1j * np.pi / 2) * (NUFIS[1][j * I_max + left_col - 1] - NUFIS[1][j * I_max + left_col])  # left boundary
    dNUFIS2[j][left_col] = (-1j * np.pi / 2) * (NUFIS[1][j * I_max + left_col - 1] - NUFIS[1][j * I_max + left_col])  # left boundary
    dNUFIS2[j][right_col - 1] = (-1j * np.pi / 2) * (NUFIS[1][j * I_max + right_col - 1] - NUFIS[1][j * I_max + right_col])  # right boundary
    dNUFIS2[j][right_col] = (-1j * np.pi / 2) * (NUFIS[1][j * I_max + right_col - 1] - NUFIS[1][j * I_max + right_col])  # right boundary
    dSIGS12[j][left_col - 1] = (-1j * np.pi / 2) * (SIGS12_reshaped[j * I_max + left_col - 1] - SIGS12_reshaped[j * I_max + left_col])  # left boundary
    dSIGS12[j][left_col] = (-1j * np.pi / 2) * (SIGS12_reshaped[j * I_max + left_col - 1] - SIGS12_reshaped[j * I_max + left_col])  # left boundary
    dSIGS12[j][right_col - 1] = (-1j * np.pi / 2) * (SIGS12_reshaped[j * I_max + right_col - 1] - SIGS12_reshaped[j * I_max + right_col])  # right boundary
    dSIGS12[j][right_col] = (-1j * np.pi / 2) * (SIGS12_reshaped[j * I_max + right_col - 1] - SIGS12_reshaped[j * I_max + right_col])  # right boundary

#    print(SIGS12_reshaped[j * I_max + left_col - 1], SIGS12_reshaped[j * I_max + left_col], dSIGS12[j][left_col], SIGS12_reshaped[j * I_max + right_col - 1], SIGS12_reshaped[j * I_max + right_col], dSIGS12[j][right_col])

dTOT = [dTOT1, dTOT2]
dNUFIS = [dNUFIS1, dNUFIS2]
v1 = [[v1] * I_max for _ in range(J_max)]
v2 = [[v2] * I_max for _ in range(J_max)]
v = [v1, v2]
dTOT_plot = [dTOT1, dTOT2]

# Reshaping
dTOT_reshaped = [[None] * N for _ in range(group)]
dNUFIS_reshaped = [[None] * N for _ in range(group)]
v_reshaped = [[None] * N for _ in range(group)]
dSIGS12_reshaped = [0.0 for _ in range(N)]
dSIGS21_reshaped = [0.0 for _ in range(N)]
dSIGS11_reshaped = [0.0 for _ in range(N)]
dSIGS22_reshaped = [0.0 for _ in range(N)]
for g in range(group):
    for j in range(J_max):  
        for i in range(I_max):
            m = j * I_max + i
            dTOT_reshaped[g][m] = dTOT[g][j][i]
            dNUFIS_reshaped[g][m] = dNUFIS[g][j][i]
            v_reshaped[g][m] = v[g][j][i]
            dSIGS11_reshaped[m] = dSIGS11[j][i]
            dSIGS12_reshaped[m] = dSIGS12[j][i]
            dSIGS21_reshaped[m] = dSIGS21[j][i]
            dSIGS22_reshaped[m] = dSIGS22[j][i]

dSIGS_reshaped = [[dSIGS11_reshaped, dSIGS21_reshaped], [dSIGS12_reshaped, dSIGS22_reshaped]]

v = v_reshaped
dTOT = dTOT_reshaped
dNUFIS = dNUFIS_reshaped

map_detector = [
9, 9, 9, 9, 1, 0, 0, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 
9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 9, 9, 
9, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 9, 
9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 9, 
0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 
0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 
0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 9, 
9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 
9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 9, 9, 
9, 9, 9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 
]

map_zone = [
99, 99, 99, 99,  1,  1,  1,  1,  2,  2,  2,  2,  2, 99, 99, 99, 99, 
99, 99,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2, 99, 99, 
99,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2, 99, 
99,  1,  1,  1,  1,  1,  4,  4,  2,  3,  3,  3,  3,  3,  3,  2, 99, 
 4,  4,  4,  4,  4,  4,  4,  4,  3,  3,  3,  3,  3,  3,  3,  3,  3, 
 4,  4,  4,  4,  4,  4,  4,  5,  5,  5,  3,  3,  3,  3,  3,  3,  3, 
 4,  4,  4,  4,  4,  6,  6,  6,  5,  5,  5,  5,  5,  5,  5,  5,  5, 
 6,  6,  6,  6,  6,  6,  6,  6,  5,  5,  5,  5,  5,  5,  5,  5,  5, 
 6,  6,  6,  6,  6,  6,  6,  6,  7,  7,  7,  7,  7,  7,  7,  7,  5, 
 6,  6,  8,  8,  8,  8,  8,  8,  7,  7,  7,  7,  7,  7,  7,  7,  7, 
 8,  8,  8,  8,  8,  8,  8,  8,  9,  9,  9,  9,  9,  7,  7,  7,  7, 
 8,  8,  8,  8,  8,  8,  8, 10,  9,  9,  9,  9,  9,  9,  9,  9,  9, 
10, 10, 10, 10, 10, 10, 10, 10, 10, 10,  9,  9,  9,  9,  9,  9,  9, 
99, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 99, 
99, 12, 12, 12, 12, 12, 12, 12, 12, 11, 11, 11, 11, 11, 11, 11, 99, 
99, 99, 12, 12, 12, 12, 12, 12, 12, 11, 11, 11, 11, 11, 11, 99, 99, 
99, 99, 99, 99, 12, 12, 12, 12, 12, 12, 11, 11, 11, 99, 99, 99, 99, 
]

# Original map_zone reshaped into 2D
map_zone_reshape = np.reshape(map_zone, (17, 17))
map_detector_reshape = np.reshape(map_detector, (17, 17))
multiplicator = np.ones((expandXS, expandXS))
map_zone_full_temp = np.kron(map_zone_reshape, multiplicator)
map_detector_full_temp = np.kron(map_detector_reshape, multiplicator)
map_zone = np.array(map_zone_full_temp).flatten().astype(int).tolist()
map_detector = np.array(map_detector_full_temp).flatten().astype(int).tolist()
