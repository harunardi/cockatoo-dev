from SRC.XSPROCESS_1D_RECT import *
from SRC.XSPROCESS_2D_RECT import *
from SRC.XSPROCESS_2D_HEXX import *
from SRC.XSPROCESS_3D_RECT import *
from SRC.XSPROCESS_3D_HEXX import *

##############################################################################
class MatrixBuilderForward1D:
    def __init__(self, group, N, TOT, SIGS_reshaped, BC, dx, D, chi, NUFIS):
        self.group = group
        self.N = N
        self.TOT = TOT
        self.SIGS_reshaped = SIGS_reshaped
        self.BC = BC
        self.dx = dx
        self.D = D
        self.chi = chi
        self.NUFIS = NUFIS

    def build_forward_matrices(self):
        D_mat = FORWARD_D_1D_matrix(self.group, self.BC, self.N, self.dx, self.D)
        TOT_mat = FORWARD_TOT_1D_matrix(self.group, self.N, self.TOT)
        SCAT_mat = FORWARD_SCAT_1D_matrix(self.group, self.N, self.SIGS_reshaped)
        F = FORWARD_NUFIS_1D_matrix(self.group, self.N, self.chi, self.NUFIS)
        M = D_mat + TOT_mat - SCAT_mat
        return M, F

class MatrixBuilderAdjoint1D:
    def __init__(self, group, N, TOT, SIGS_reshaped, BC, dx, D, chi, NUFIS):
        self.group = group
        self.N = N
        self.TOT = TOT
        self.SIGS_reshaped = SIGS_reshaped
        self.BC = BC
        self.dx = dx
        self.D = D
        self.chi = chi
        self.NUFIS = NUFIS

    def build_adjoint_matrices(self):
        D_mat = ADJOINT_D_1D_matrix(self.group, self.BC, self.N, self.dx, self.D)
        TOT_mat = ADJOINT_TOT_1D_matrix(self.group, self.N, self.TOT)
        SCAT_mat = ADJOINT_SCAT_1D_matrix(self.group, self.N, self.SIGS_reshaped)
        F = ADJOINT_NUFIS_1D_matrix(self.group, self.N, self.chi, self.NUFIS)
        M = D_mat + TOT_mat - SCAT_mat
        return M, F

class MatrixBuilderNoise1D:
    def __init__(self, group, N, TOT, SIGS_reshaped, BC, dx, D, chi, NUFIS, keff, v, Beff, omega, l, dTOT, dSIGS_reshaped, dNUFIS):
        self.group = group
        self.N = N
        self.TOT = TOT
        self.SIGS_reshaped = SIGS_reshaped
        self.BC = BC
        self.dx = dx
        self.D = D
        self.chi = chi
        self.NUFIS = NUFIS
        self.keff = keff
        self.v = v
        self.Beff = Beff
        self.omega = omega
        self.l = l
        self.dTOT = dTOT
        self.dSIGS_reshaped = dSIGS_reshaped
        self.dNUFIS = dNUFIS

    def build_noise_matrices(self):
        chi_p = self.chi
        chi_d = self.chi
        k_complex = 1/self.keff* ((self.l * self.Beff) / (self.l + 1j * self.omega))
        D_mat = NOISE_D_1D_matrix(self.group, self.BC, self.N, self.dx, self.D)
        TOT_mat = NOISE_TOT_1D_matrix(self.group, self.N, self.TOT)
        SCAT_mat = NOISE_SCAT_1D_matrix(self.group, self.N, self.SIGS_reshaped)
        NUFIS_mat = NOISE_NUFIS_1D_matrix(self.group, self.N, chi_p, chi_d, self.NUFIS, k_complex, self.Beff, self.keff)
        FREQ_mat = NOISE_FREQ_1D_matrix(self.group, self.N, self.omega, self.v)
        M = FREQ_mat - D_mat + TOT_mat - NUFIS_mat - SCAT_mat
        dTOT_mat = NOISE_dTOT_1D_matrix(self.group, self.N, self.dTOT)
        dSCAT_mat = NOISE_dSCAT_1D_matrix(self.group, self.N, self.dSIGS_reshaped)
        dNUFIS_mat = NOISE_dNUFIS_1D_matrix(self.group, self.N, chi_p, chi_d, self.dNUFIS, k_complex, self.Beff, self.keff)
        dS = -dTOT_mat + dSCAT_mat + dNUFIS_mat
        return M, dS

##############################################################################
class MatrixBuilderForward2DRect:
    def __init__(self, group, N, conv, TOT, SIGS_reshaped, BC, dx, dy, D, chi, NUFIS):
        self.group = group
        self.N = N
        self.conv = conv
        self.TOT = TOT
        self.SIGS_reshaped = SIGS_reshaped
        self.BC = BC
        self.dx = dx
        self.dy = dy
        self.D = D
        self.chi = chi
        self.NUFIS = NUFIS

    def build_forward_matrices(self):
        D_mat = FORWARD_D_2D_rect_matrix(self.group, self.BC, self.conv, self.dx, self.dy, self.D)
        TOT_mat = FORWARD_TOT_2D_rect_matrix(self.group, self.N, self.conv, self.TOT)
        SCAT_mat = FORWARD_SCAT_2D_rect_matrix(self.group, self.N, self.conv, self.SIGS_reshaped)
        M = D_mat + TOT_mat - SCAT_mat
        F = FORWARD_NUFIS_2D_rect_matrix(self.group, self.N, self.conv, self.chi, self.NUFIS)
        return M.tocsr(), F.tocsr()

class MatrixBuilderAdjoint2DRect:
    def __init__(self, group, N, conv, TOT, SIGS_reshaped, BC, dx, dy, D, chi, NUFIS):
        self.group = group
        self.N = N
        self.conv = conv
        self.TOT = TOT
        self.SIGS_reshaped = SIGS_reshaped
        self.BC = BC
        self.dx = dx
        self.dy = dy
        self.D = D
        self.chi = chi
        self.NUFIS = NUFIS

    def build_adjoint_matrices(self):
        TOT_mat = ADJOINT_TOT_2D_rect_matrix(self.group, self.N, self.conv, self.TOT)
        SCAT_mat = ADJOINT_SCAT_2D_rect_matrix(self.group, self.N, self.conv, self.SIGS_reshaped)
        D_mat = ADJOINT_D_2D_rect_matrix(self.group, self.BC, self.conv, self.dx, self.dy, self.D)
        F = ADJOINT_NUFIS_2D_rect_matrix(self.group, self.N, self.conv, self.chi, self.NUFIS)
        M = D_mat + TOT_mat - SCAT_mat
        return M.tocsr(), F.tocsr()

class MatrixBuilderNoise2DRect:
    def __init__(self, group, N, conv, TOT, SIGS_reshaped, BC, dx, dy, D, chi, NUFIS, keff, v, Beff, omega, l, dTOT, dSIGS_reshaped, dNUFIS):
        self.group = group
        self.N = N
        self.conv = conv
        self.TOT = TOT
        self.SIGS_reshaped = SIGS_reshaped
        self.BC = BC
        self.dx = dx
        self.dy = dy
        self.D = D
        self.chi = chi
        self.NUFIS = NUFIS
        self.keff = keff
        self.v = v
        self.Beff = Beff
        self.omega = omega
        self.l = l
        self.dTOT = dTOT
        self.dSIGS_reshaped = dSIGS_reshaped
        self.dNUFIS = dNUFIS

    def build_noise_matrices(self):
        chi_p = self.chi
        chi_d = self.chi
        k_complex = 1/self.keff* ((self.l * self.Beff) / (self.l + 1j * self.omega))
        D_mat = NOISE_D_2D_rect_matrix(self.group, self.BC, self.conv, self.dx, self.dy, self.D)
        TOT_mat = NOISE_TOT_2D_rect_matrix(self.group, self.N, self.conv, self.TOT)
        FREQ_mat = NOISE_FREQ_2D_rect_matrix(self.group, self.N, self.conv, self.omega, self.v)
        SCAT_mat = NOISE_SCAT_2D_rect_matrix(self.group, self.N, self.conv, self.SIGS_reshaped)
        NUFIS_mat = NOISE_NUFIS_2D_rect_matrix(self.group, self.N, self.conv, chi_p, chi_d, self.NUFIS, k_complex, self.Beff, self.keff)
        M = FREQ_mat - D_mat + TOT_mat - NUFIS_mat - SCAT_mat
        dTOT_mat = NOISE_dTOT_2D_rect_matrix(self.group, self.N, self.conv, self.dTOT)
        dSCAT_mat = NOISE_dSCAT_2D_rect_matrix(self.group, self.N, self.conv, self.dSIGS_reshaped)
        dNUFIS_mat = NOISE_dNUFIS_2D_rect_matrix(self.group, self.N, self.conv, chi_p, chi_d, self.dNUFIS, k_complex, self.Beff, self.keff)
        dS = -dTOT_mat + dSCAT_mat + dNUFIS_mat
        return M.tocsr(), dS

##############################################################################
class MatrixBuilderForward2DHexx:
    def __init__(self, group, I_max, J_max, conv_tri, conv_neighbor, TOT, SIGS_reshaped, BC, h, level, D, chi, NUFIS):
        self.group = group
        self.I_max = I_max
        self.J_max = J_max
        self.conv_tri = conv_tri
        self.conv_neighbor = conv_neighbor
        self.TOT = TOT
        self.SIGS_reshaped = SIGS_reshaped
        self.BC = BC
        self.h = h
        self.level = level
        self.D = D
        self.chi = chi
        self.NUFIS = NUFIS

    def build_forward_matrices(self):
        D_hexx_mat = FORWARD_D_2D_hexx_matrix(self.group, self.BC, self.conv_tri, self.conv_neighbor, self.h, self.D, self.level)
        TOT_mat = FORWARD_TOT_2D_hexx_matrix(self.group, self.I_max, self.J_max, self.conv_tri, self.TOT, self.level)
        SCAT_mat = FORWARD_SCAT_2D_hexx_matrix(self.group, self.I_max, self.J_max, self.conv_tri, self.SIGS_reshaped, self.level)
        M = D_hexx_mat + TOT_mat - SCAT_mat
        F = FORWARD_NUFIS_2D_hexx_matrix(self.group, self.I_max, self.J_max, self.conv_tri, self.chi, self.NUFIS, self.level)
        return M.tocsr(), F.tocsr()

class MatrixBuilderAdjoint2DHexx:
    def __init__(self, group, I_max, J_max, conv_tri, conv_neighbor, TOT, SIGS_reshaped, BC, h, level, D, chi, NUFIS):
        self.group = group
        self.I_max = I_max
        self.J_max = J_max
        self.conv_tri = conv_tri
        self.conv_neighbor = conv_neighbor
        self.TOT = TOT
        self.SIGS_reshaped = SIGS_reshaped
        self.BC = BC
        self.h = h
        self.level = level
        self.D = D
        self.chi = chi
        self.NUFIS = NUFIS

    def build_adjoint_matrices(self):
        D_hexx_mat = ADJOINT_D_2D_hexx_matrix(self.group, self.BC, self.conv_tri, self.conv_neighbor, self.h, self.D, self.level)
        TOT_mat = ADJOINT_TOT_2D_hexx_matrix(self.group, self.I_max, self.J_max, self.conv_tri, self.TOT, self.level)
        SCAT_mat = ADJOINT_SCAT_2D_hexx_matrix(self.group, self.I_max, self.J_max, self.conv_tri, self.SIGS_reshaped, self.level)
        M = D_hexx_mat + TOT_mat - SCAT_mat
        F = ADJOINT_NUFIS_2D_hexx_matrix(self.group, self.I_max, self.J_max, self.conv_tri, self.chi, self.NUFIS, self.level)
        return M.tocsr(), F.tocsr()

class MatrixBuilderNoise2DHexx:
    def __init__(self, group, I_max, J_max, N_hexx, conv_tri, conv_neighbor, TOT, SIGS_reshaped, BC, h, level, D, chi, NUFIS, keff, v, Beff, omega, l, dTOT_hexx, dSIGS_hexx, chi_hexx, dNUFIS_hexx, noise_section, type_noise):
        self.group = group
        self.I_max = I_max
        self.J_max = J_max
        self.conv_tri = conv_tri
        self.conv_neighbor = conv_neighbor
        self.TOT = TOT
        self.SIGS_reshaped = SIGS_reshaped
        self.BC = BC
        self.h = h
        self.level = level
        self.D = D
        self.chi = chi
        self.NUFIS = NUFIS
        self.keff = keff
        self.v = v
        self.Beff = Beff
        self.omega = omega
        self.l = l
        self.dTOT_hexx = dTOT_hexx
        self.dSIGS_hexx = dSIGS_hexx
        self.chi_hexx = chi_hexx
        self.dNUFIS_hexx = dNUFIS_hexx
        self.noise_section = noise_section
        self.type_noise = type_noise
        self.N_hexx = N_hexx

    def build_noise_matrices(self):
        chi_p = self.chi
        chi_d = self.chi
        chi_p_hexx = self.chi_hexx
        chi_d_hexx = self.chi_hexx
        k_complex = 1/self.keff* ((self.l * self.Beff) / (self.l + 1j * self.omega))
        D_hexx_mat = NOISE_D_2D_hexx_matrix(self.group, self.BC, self.conv_tri, self.conv_neighbor, self.h, self.D, self.level)
        TOT_mat = NOISE_TOT_2D_hexx_matrix(self.group, self.I_max, self.J_max, self.conv_tri, self.TOT, self.level)
        FREQ_mat = NOISE_FREQ_2D_hexx_matrix(self.group, self.I_max, self.J_max, self.conv_tri, self.omega, self.v, self.level)
        SCAT_mat = NOISE_SCAT_2D_hexx_matrix(self.group, self.I_max, self.J_max, self.conv_tri, self.SIGS_reshaped, self.level)
        NUFIS_mat = NOISE_NUFIS_2D_hexx_matrix(self.group, self.I_max, self.J_max, self.conv_tri, chi_p, chi_d, self.NUFIS, k_complex, self.Beff, self.keff, self.level)
        M = FREQ_mat - D_hexx_mat + TOT_mat - NUFIS_mat - SCAT_mat

        dTOT_mat = NOISE_dTOT_2D_hexx_matrix(self.group, self.I_max, self.J_max, self.conv_tri, self.dTOT_hexx, self.level)
        dSCAT_mat = NOISE_dSCAT_2D_hexx_matrix(self.group, self.I_max, self.J_max, self.conv_tri, self.dSIGS_hexx, self.level)
        dNUFIS_mat = NOISE_dNUFIS_2D_hexx_matrix(self.group, self.I_max, self.J_max, self.conv_tri, chi_p_hexx, chi_d_hexx, self.dNUFIS_hexx, k_complex, self.Beff, self.keff, self.level)
        dS = -dTOT_mat + dSCAT_mat + dNUFIS_mat
        return M.tocsr(), dS

##############################################################################
class MatrixBuilderForward3DRect:
    def __init__(self, group, N, conv, TOT, SIGS_reshaped, BC, dx, dy, dz, D, chi, NUFIS):
        self.group = group
        self.N = N
        self.conv = conv
        self.TOT = TOT
        self.SIGS_reshaped = SIGS_reshaped
        self.BC = BC
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.D = D
        self.chi = chi
        self.NUFIS = NUFIS

    def build_forward_matrices(self):
        D_mat = FORWARD_D_3D_rect_matrix(self.group, self.BC, self.conv, self.dx, self.dy, self.dz, self.D)
        TOT_mat = FORWARD_TOT_3D_rect_matrix(self.group, self.N, self.conv, self.TOT)
        SCAT_mat = FORWARD_SCAT_3D_rect_matrix(self.group, self.N, self.conv, self.SIGS_reshaped)
        M = D_mat + TOT_mat - SCAT_mat
        F = FORWARD_NUFIS_3D_rect_matrix(self.group, self.N, self.conv, self.chi, self.NUFIS)
        return M.tocsr(), F.tocsr()

class MatrixBuilderAdjoint3DRect:
    def __init__(self, group, N, conv, TOT, SIGS_reshaped, BC, dx, dy, dz, D, chi, NUFIS):
        self.group = group
        self.N = N
        self.conv = conv
        self.TOT = TOT
        self.SIGS_reshaped = SIGS_reshaped
        self.BC = BC
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.D = D
        self.chi = chi
        self.NUFIS = NUFIS

    def build_adjoint_matrices(self):
        TOT_mat = ADJOINT_TOT_3D_rect_matrix(self.group, self.N, self.conv, self.TOT)
        SCAT_mat = ADJOINT_SCAT_3D_rect_matrix(self.group, self.N, self.conv, self.SIGS_reshaped)
        D_mat = ADJOINT_D_3D_rect_matrix(self.group, self.BC, self.conv, self.dx, self.dy, self.dz, self.D)
        F = ADJOINT_NUFIS_3D_rect_matrix(self.group, self.N, self.conv, self.chi, self.NUFIS)
        M = D_mat + TOT_mat - SCAT_mat
        return M.tocsr(), F.tocsr()

class MatrixBuilderNoise3DRect:
    def __init__(self, group, N, conv, TOT, SIGS_reshaped, BC, dx, dy, dz, D, chi, NUFIS, keff, v, Beff, omega, l, dTOT, dSIGS_reshaped, dNUFIS):
        self.group = group
        self.N = N
        self.conv = conv
        self.TOT = TOT
        self.SIGS_reshaped = SIGS_reshaped
        self.BC = BC
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.D = D
        self.chi = chi
        self.NUFIS = NUFIS
        self.keff = keff
        self.v = v
        self.Beff = Beff
        self.omega = omega
        self.l = l
        self.dTOT = dTOT
        self.dSIGS_reshaped = dSIGS_reshaped
        self.dNUFIS = dNUFIS

    def build_noise_matrices(self):
        chi_p = self.chi
        chi_d = self.chi
        k_complex = 1/self.keff* ((self.l * self.Beff) / (self.l + 1j * self.omega))
        D_mat = NOISE_D_3D_rect_matrix(self.group, self.BC, self.conv, self.dx, self.dy, self.dz, self.D)
        TOT_mat = NOISE_TOT_3D_rect_matrix(self.group, self.N, self.conv, self.TOT)
        FREQ_mat = NOISE_FREQ_3D_rect_matrix(self.group, self.N, self.conv, self.omega, self.v)
        SCAT_mat = NOISE_SCAT_3D_rect_matrix(self.group, self.N, self.conv, self.SIGS_reshaped)
        NUFIS_mat = NOISE_NUFIS_3D_rect_matrix(self.group, self.N, self.conv, chi_p, chi_d, self.NUFIS, k_complex, self.Beff, self.keff)
        M = FREQ_mat - D_mat + TOT_mat - NUFIS_mat - SCAT_mat
        dTOT_mat = NOISE_dTOT_3D_rect_matrix(self.group, self.N, self.conv, self.dTOT)
        dSCAT_mat = NOISE_dSCAT_3D_rect_matrix(self.group, self.N, self.conv, self.dSIGS_reshaped)
        dNUFIS_mat = NOISE_dNUFIS_3D_rect_matrix(self.group, self.N, self.conv, chi_p, chi_d, self.dNUFIS, k_complex, self.Beff, self.keff)
        dS = -dTOT_mat + dSCAT_mat + dNUFIS_mat
        return M.tocsr(), dS

##############################################################################
class MatrixBuilderForward3DHexx:
    def __init__(self, group, I_max, J_max, K_max, conv_tri, conv_neighbor_3D, TOT, SIGS_reshaped, BC, h, dz, level, D, chi, NUFIS):
        self.group = group
        self.I_max = I_max
        self.J_max = J_max
        self.K_max = K_max
        self.conv_tri = conv_tri
        self.conv_neighbor_3D = conv_neighbor_3D
        self.TOT = TOT
        self.SIGS_reshaped = SIGS_reshaped
        self.BC = BC
        self.h = h
        self.dz = dz
        self.level = level
        self.D = D
        self.chi = chi
        self.NUFIS = NUFIS

    def build_forward_matrices(self):
        D_hexx_mat = FORWARD_D_3D_hexx_matrix(self.group, self.BC, self.conv_tri, self.conv_neighbor_3D, self.h, self.dz, self.D, self.level)
        TOT_mat = FORWARD_TOT_3D_hexx_matrix(self.group, self.K_max, self.J_max, self.I_max, self.conv_tri, self.TOT, self.level)
        SCAT_mat = FORWARD_SCAT_3D_hexx_matrix(self.group, self.K_max, self.J_max, self.I_max, self.conv_tri, self.SIGS_reshaped, self.level)
        M = D_hexx_mat + TOT_mat - SCAT_mat
        F = FORWARD_NUFIS_3D_hexx_matrix(self.group, self.K_max, self.J_max, self.I_max, self.conv_tri, self.chi, self.NUFIS, self.level)
        return M.tocsr(), F.tocsr()

class MatrixBuilderAdjoint3DHexx:
    def __init__(self, group, I_max, J_max, K_max, conv_tri, conv_neighbor_3D, TOT, SIGS_reshaped, BC, h, dz, level, D, chi, NUFIS):
        self.group = group
        self.I_max = I_max
        self.J_max = J_max
        self.K_max = K_max
        self.conv_tri = conv_tri
        self.conv_neighbor_3D = conv_neighbor_3D
        self.TOT = TOT
        self.SIGS_reshaped = SIGS_reshaped
        self.BC = BC
        self.h = h
        self.dz = dz
        self.level = level
        self.D = D
        self.chi = chi
        self.NUFIS = NUFIS

    def build_adjoint_matrices(self):
        D_hexx_mat = ADJOINT_D_3D_hexx_matrix(self.group, self.BC, self.conv_tri, self.conv_neighbor_3D, self.h, self.dz, self.D, self.level)
        TOT_mat = ADJOINT_TOT_3D_hexx_matrix(self.group, self.K_max, self.J_max, self.I_max, self.conv_tri, self.TOT, self.level)
        SCAT_mat = ADJOINT_SCAT_3D_hexx_matrix(self.group, self.K_max, self.J_max, self.I_max, self.conv_tri, self.SIGS_reshaped, self.level)
        M = D_hexx_mat + TOT_mat - SCAT_mat
        F = ADJOINT_NUFIS_3D_hexx_matrix(self.group, self.K_max, self.J_max, self.I_max, self.conv_tri, self.chi, self.NUFIS, self.level)
        return M.tocsr(), F.tocsr()

class MatrixBuilderNoise3DHexx:
    def __init__(self, group, I_max, J_max, K_max, N_hexx, conv_tri, conv_neighbor_3D, TOT, SIGS_reshaped, BC, h, dz, level, D, chi, NUFIS, keff, v, Beff, omega, l, dTOT_hexx, dSIGS_hexx, chi_hexx, dNUFIS_hexx, noise_section, type_noise):
        self.group = group
        self.I_max = I_max
        self.J_max = J_max
        self.K_max = K_max
        self.conv_tri = conv_tri
        self.conv_neighbor_3D = conv_neighbor_3D
        self.TOT = TOT
        self.SIGS_reshaped = SIGS_reshaped
        self.BC = BC
        self.h = h
        self.dz = dz
        self.level = level
        self.D = D
        self.chi = chi
        self.NUFIS = NUFIS
        self.keff = keff
        self.v = v
        self.Beff = Beff
        self.omega = omega
        self.l = l
        self.dTOT_hexx = dTOT_hexx
        self.dSIGS_hexx = dSIGS_hexx
        self.chi_hexx = chi_hexx
        self.dNUFIS_hexx = dNUFIS_hexx
        self.noise_section = noise_section
        self.type_noise = type_noise
        self.N_hexx = N_hexx

    def build_noise_matrices(self):
        chi_p = self.chi
        chi_d = self.chi
        chi_p_hexx = self.chi_hexx
        chi_d_hexx = self.chi_hexx
        k_complex = 1/self.keff* ((self.l * self.Beff) / (self.l + 1j * self.omega))
        D_hexx_mat = NOISE_D_3D_hexx_matrix(self.group, self.BC, self.conv_tri, self.conv_neighbor_3D, self.h, self.dz, self.D, self.level)
        TOT_mat = NOISE_TOT_3D_hexx_matrix(self.group, self.K_max, self.J_max, self.I_max, self.conv_tri, self.TOT, self.level)
        FREQ_mat = NOISE_FREQ_3D_hexx_matrix(self.group, self.K_max, self.J_max, self.I_max, self.conv_tri, self.omega, self.v, self.level)
        SCAT_mat = NOISE_SCAT_3D_hexx_matrix(self.group, self.K_max, self.J_max, self.I_max, self.conv_tri, self.SIGS_reshaped, self.level)
        NUFIS_mat = NOISE_NUFIS_3D_hexx_matrix(self.group, self.K_max, self.J_max, self.I_max, self.conv_tri, chi_p, chi_d, self.NUFIS, k_complex, self.Beff, self.keff, self.level)
        M = FREQ_mat - D_hexx_mat + TOT_mat - NUFIS_mat - SCAT_mat

        dTOT_mat = NOISE_dTOT_3D_hexx_matrix(self.group, self.K_max, self.J_max, self.I_max, self.conv_tri, self.dTOT_hexx, self.level)
        dSCAT_mat = NOISE_dSCAT_3D_hexx_matrix(self.group, self.K_max, self.J_max, self.I_max, self.conv_tri, self.dSIGS_hexx, self.level)
        dNUFIS_mat = NOISE_dNUFIS_3D_hexx_matrix(self.group, self.K_max, self.J_max, self.I_max, self.conv_tri, chi_p_hexx, chi_d_hexx, self.dNUFIS_hexx, k_complex, self.Beff, self.keff, self.level)
        dS = -dTOT_mat + dSCAT_mat + dNUFIS_mat
        return M.tocsr(), dS
