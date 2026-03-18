import numpy as np
from scipy.sparse.linalg import spilu, LinearOperator, spsolve, gmres, splu, cg
from scipy.integrate import trapezoid
from petsc4py import PETSc

class PowerMethodSolver1D:
    def __init__(self, group, N, M, F, x, precond, tol=1e-06):
        self.group = group
        self.N = N
        self.M = M
        self.F = F
        self.x = x
        self.precond = precond
        self.tol = tol

    def solve(self):
        phi = np.ones(self.group * self.N)
        keff = 1.0
        errflux = errkeff = self.tol + 1
        iter_count = 0
        x_integrate = np.tile(self.x, self.group)

        if self.precond == 1:
            print(f'Solving using ILU')
            M_csc = self.M.tocsc()
            ilu = spilu(M_csc)
            M_preconditioner = LinearOperator(M_csc.shape, matvec=ilu.solve)
        elif self.precond == 2:
            print('Solving using LU Decomposition')
            M_csc = self.M.tocsc()
            lu = splu(M_csc)
            M_preconditioner = LinearOperator(M_csc.shape, matvec=lu.solve)
        else:
            print(f'Solving using Sparse Solver')

        while errflux > self.tol:
            phi_old = phi.copy()
            k_old = keff

            S = 1 / k_old * (self.F @ phi_old)
            if self.precond == 1:
                phi, info = gmres(M_csc, S, rtol=1e-8, maxiter=1000, M=M_preconditioner)
            elif self.precond == 2:
                phi = lu.solve(S)
            else:
                phi = spsolve(self.M, S)

            # Update keff
            keff = k_old * trapezoid(self.F @ phi, x=x_integrate, axis=0) / \
                   trapezoid(self.F @ phi_old, x=x_integrate, axis=0)

            residual = S - self.M.dot(phi)
            residual_norm = np.linalg.norm(residual)

            # Normalization
            phi /= np.max(phi)

            # Calculate errors
            errkeff = np.abs((keff - k_old) / k_old)
            errflux = np.max(np.abs(phi - phi_old) / (np.abs(phi) + 1E-20))

            iter_count += 1
            print(f'Iteration: {iter_count}, keff = {keff:.5f}, errkeff = {errkeff:.6e}, '
                  f'errflux = {errflux:.5e}, residual = {residual_norm:.5e}')

        return keff, phi

class FixedSourceSolver1D:
    def __init__(self, group, N, M, dS, dSOURCE, PHI, precond, tol=1e-06):
        self.group = group
        self.N = N
        self.M = M
        self.dS = dS
        self.precond = precond
        self.tol = tol
        self.PHI = PHI
        self.dSOURCE = dSOURCE

    def solve(self):
        dPHI = np.ones(self.group * self.N, dtype=complex)
        errdPHI = 1
        tol = 1e-06
        iter = 0
        self.dSOURCE = [item for sublist in self.dSOURCE for item in sublist] if all(isinstance(sublist, list) for sublist in self.dSOURCE) else self.dSOURCE #[item for sublist in self.dSOURCE for item in sublist]

        if self.precond == 1:
            print('Solving using ILU')
            M_csc = self.M.tocsc()
            ilu = spilu(M_csc)
            M_preconditioner = LinearOperator(M_csc.shape, matvec=ilu.solve)
        elif self.precond == 2:
            print('Solving using LU Decomposition')
            M_csc = self.M.tocsc()
            lu = splu(M_csc)
            M_preconditioner = LinearOperator(M_csc.shape, matvec=lu.solve)
        else:
            print('Solving using Solver')

        while errdPHI > self.tol:
            dPHI_old = dPHI.copy()

            # Set up RHS
            S = self.dS.dot(self.PHI) + self.dSOURCE

            if self.precond == 1:
                dPHI, info = cg(M_csc, S, rtol=1e-8, maxiter=1000, M=M_preconditioner)
            elif self.precond == 2:
                dPHI = lu.solve(S)
            else:
                dPHI = spsolve(self.M, S)

            # Calculate errors
            errdPHI = np.max(np.abs(dPHI - dPHI_old) / (np.abs(dPHI) + 1E-20))

            iter += 1
            print(f'Iteration: {iter}, errflux = {errdPHI:.6e}')

        return dPHI

class PowerMethodSolver2DRect:
    def __init__(self, group, N, conv, M, F, dx, dy, precond, tol):
        self.group = group
        self.N = N
        self.M = M
        self.F = F
        self.dx = dx
        self.dy = dy
        self.tol = tol
        self.precond = precond
        self.conv = conv

    def solve(self):
        M_petsc = PETSc.Mat().createAIJ(size=self.M.shape, csr=(self.M.indptr, self.M.indices, self.M.data))
        F_petsc = PETSc.Mat().createAIJ(size=self.F.shape, csr=(self.F.indptr, self.F.indices, self.F.data))
        M_petsc.assemble()
        F_petsc.assemble()

        # PETSc Solver (KSP) and Preconditioner (PC)
        ksp = PETSc.KSP().create()
        ksp.setOperators(M_petsc)
        ksp.setType(PETSc.KSP.Type.GMRES)

        # Preconditioner setup
        pc = ksp.getPC()
        if self.precond == 0:
            print(f'Solving using Sparse Solver')
            pc.setType(PETSc.PC.Type.NONE)
        elif self.precond == 1:
            print(f'Solving using ILU')
            pc.setType(PETSc.PC.Type.ILU)
            print(f'ILU Preconditioner Done')
        elif self.precond == 2:
            print('Solving using LU Decomposition')
            pc.setType(PETSc.PC.Type.LU)
            print(f'LU Preconditioner Done')

        # Solver tolerances
        ksp.setTolerances(rtol=1e-10, max_it=5000)

        # Iteration for Power Method
        phi_temp = PETSc.Vec().createSeq(self.group * max(self.conv))
        phi_temp.set(1.0)
        keff = 1.0
        errflux = errkeff = 1.0
        iter_count = 0

        S = phi_temp.duplicate()
        Fphi_old = phi_temp.duplicate()
        Fphi_new = phi_temp.duplicate()

        w = phi_temp.duplicate()
        vol = self.dx * self.dy
        w.set(vol)

        while errflux > self.tol and errkeff > self.tol:
            phi_temp_old = phi_temp.copy()
            k_old = keff

            # S = 1 / k_old * (self.F @ phi_temp_old)
            F_petsc.mult(phi_temp_old, S)
            S.scale(1 / k_old)

            # Solve the linear system using PETSc KSP
            ksp.solve(S, phi_temp)

            F_petsc.mult(phi_temp_old, Fphi_old)
            F_petsc.mult(phi_temp, Fphi_new)

            # Update keff
            num = Fphi_new.dot(w)
            den = Fphi_old.dot(w)
            keff = k_old * num / den

            # Normalization
            _, max_val = phi_temp.max()
            phi_temp.scale(1.0/max_val)

            # Calculate errors
            phi_diff = phi_temp.copy()
            phi_diff.axpy(-1.0, phi_temp_old)
            errflux = phi_diff.norm() / (phi_temp.norm() + 1E-20)
            errkeff = abs((keff - k_old) / k_old)

            iter_count += 1
            print(f'Iteration: {iter_count}, keff = {keff:.5f}, errkeff = {errkeff:.6e}, '
                  f'errflux = {errflux:.5e}')

        return keff, phi_temp.getArray().copy()

class FixedSourceSolver2DRect:
    def __init__(self, group, N, conv, M, dS, PHI, dx, dy, precond, tol):
        self.group = group
        self.N = N
        self.M = M
        self.dS = dS
        self.dx = dx
        self.dy = dy
        self.tol = tol
        self.conv = conv
        self.PHI = PHI
        self.precond = precond

    def solve(self):
        M_petsc = PETSc.Mat().createAIJ(size=self.M.shape, csr=(self.M.indptr, self.M.indices, self.M.data), comm=PETSc.COMM_WORLD)
        dS_petsc = PETSc.Mat().createAIJ(size=self.dS.shape, csr=(self.dS.indptr, self.dS.indices, self.dS.data), comm=PETSc.COMM_WORLD)
        M_petsc.assemble()
        dS_petsc.assemble()

        PHI_vec = PETSc.Vec().createSeq(self.dS.shape[1])
        PHI_vec.setArray(self.PHI)

        S = PETSc.Vec().createSeq(self.dS.shape[0])
        dS_petsc.mult(PHI_vec, S)

        # PETSc Solver (KSP) and Preconditioner (PC)
        ksp = PETSc.KSP().create()
        ksp.setOperators(M_petsc)
        ksp.setType(PETSc.KSP.Type.GMRES)

        # Preconditioner setup
        pc = ksp.getPC()
        if self.precond == 0:
            print(f'Solving using Sparse Solver')
            pc.setType(PETSc.PC.Type.NONE)
        elif self.precond == 1:
            print(f'Solving using ILU')
            pc.setType(PETSc.PC.Type.ILU)
            print(f'ILU Preconditioner Done')
        elif self.precond == 2:
            print('Solving using LU Decomposition')
            pc.setType(PETSc.PC.Type.LU)
            print(f'LU Preconditioner Done')

        # Solver tolerances
        ksp.setTolerances(rtol=1e-10, max_it=5000)

        # Iteration for Power Method
        dPHI_temp = PETSc.Vec().createSeq(self.group * max(self.conv))
        dPHI_temp.set(1.0+0j)
        errdPHI = 1.0
        iter_count = 0

        while errdPHI > self.tol:
            dPHI_temp_old = dPHI_temp.copy()

            # Solve the linear system using PETSc KSP
            ksp.solve(S, dPHI_temp)

            # Calculate errors
            dPHI_diff = dPHI_temp.copy()
            dPHI_diff.axpy(-1.0, dPHI_temp_old)
            errdPHI = dPHI_diff.norm() / (dPHI_temp.norm() + 1E-20)

            iter_count += 1
            print(f'Iteration: {iter_count}, errflux = {errdPHI:.5e}')

        return dPHI_temp.getArray().copy()

class PowerMethodSolver2DHexx:
    def __init__(self, group, conv_tri, M, F, h, precond, tol):
        self.group = group
        self.M = M
        self.F = F
        self.h = h
        self.tol = tol
        self.precond = precond
        self.conv_tri = conv_tri

    def solve(self):
        M_petsc = PETSc.Mat().createAIJ(size=self.M.shape, csr=(self.M.indptr, self.M.indices, self.M.data))
        F_petsc = PETSc.Mat().createAIJ(size=self.F.shape, csr=(self.F.indptr, self.F.indices, self.F.data))
        M_petsc.assemble()
        F_petsc.assemble()

        # PETSc Solver (KSP) and Preconditioner (PC)
        ksp = PETSc.KSP().create()
        ksp.setOperators(M_petsc)
        ksp.setType(PETSc.KSP.Type.GMRES)

        # Preconditioner setup
        pc = ksp.getPC()
        if self.precond == 0:
            print(f'Solving using Sparse Solver')
            pc.setType(PETSc.PC.Type.NONE)
        elif self.precond == 1:
            print(f'Solving using ILU')
            pc.setType(PETSc.PC.Type.ILU)
            print(f'ILU Preconditioner Done')
        elif self.precond == 2:
            print('Solving using LU Decomposition')
            pc.setType(PETSc.PC.Type.LU)
            print(f'LU Preconditioner Done')

        # Solver tolerances
        ksp.setTolerances(rtol=1e-10, max_it=5000)

        # Iteration for Power Method
        phi_temp = PETSc.Vec().createSeq(self.group * max(self.conv_tri))
        phi_temp.set(1.0)
        keff = 1.0
        errflux = errkeff = 1.0
        iter_count = 0

        S = phi_temp.duplicate()
        Fphi_old = phi_temp.duplicate()
        Fphi_new = phi_temp.duplicate()

        w = phi_temp.duplicate()
        vol = self.h**2/4*np.sqrt(3)
        w.set(vol)

        while errflux > self.tol and errkeff > self.tol:
            phi_temp_old = phi_temp.copy()
            k_old = keff

            # S = 1 / k_old * (self.F @ phi_temp_old)
            F_petsc.mult(phi_temp_old, S)
            S.scale(1 / k_old)

            # Solve the linear system using PETSc KSP
            ksp.solve(S, phi_temp)

            F_petsc.mult(phi_temp_old, Fphi_old)
            F_petsc.mult(phi_temp, Fphi_new)

            # Update keff
            num = Fphi_new.dot(w)
            den = Fphi_old.dot(w)
            keff = k_old * num / den

            # Normalization
            _, max_val = phi_temp.max()
            phi_temp.scale(1.0/max_val)

            # Calculate errors
            phi_diff = phi_temp.copy()
            phi_diff.axpy(-1.0, phi_temp_old)
            errflux = phi_diff.norm() / (phi_temp.norm() + 1E-20)
            errkeff = abs((keff - k_old) / k_old)

            iter_count += 1
            print(f'Iteration: {iter_count}, keff = {keff:.5f}, errkeff = {errkeff:.6e}, '
                  f'errflux = {errflux:.5e}')

        return keff, phi_temp.getArray().copy()

class FixedSourceSolver2DHexx:
    def __init__(self, group, conv_tri, M, dS, PHI, precond, tol):
        self.group = group
        self.M = M
        self.dS = dS
        self.tol = tol
        self.conv_tri = conv_tri
        self.PHI = PHI
        self.precond = precond

    def solve(self):
        M_petsc = PETSc.Mat().createAIJ(size=self.M.shape, csr=(self.M.indptr, self.M.indices, self.M.data), comm=PETSc.COMM_WORLD)
        dS_petsc = PETSc.Mat().createAIJ(size=self.dS.shape, csr=(self.dS.indptr, self.dS.indices, self.dS.data), comm=PETSc.COMM_WORLD)
        M_petsc.assemble()
        dS_petsc.assemble()

        PHI_vec = PETSc.Vec().createSeq(self.dS.shape[1])
        PHI_vec.setArray(self.PHI)

        S = PETSc.Vec().createSeq(self.dS.shape[0])
        dS_petsc.mult(PHI_vec, S)

        # PETSc Solver (KSP) and Preconditioner (PC)
        ksp = PETSc.KSP().create()
        ksp.setOperators(M_petsc)
        ksp.setType(PETSc.KSP.Type.GMRES)

        # Preconditioner setup
        pc = ksp.getPC()
        if self.precond == 0:
            print(f'Solving using Sparse Solver')
            pc.setType(PETSc.PC.Type.NONE)
        elif self.precond == 1:
            print(f'Solving using ILU')
            pc.setType(PETSc.PC.Type.ILU)
            print(f'ILU Preconditioner Done')
        elif self.precond == 2:
            print('Solving using LU Decomposition')
            pc.setType(PETSc.PC.Type.LU)
            print(f'LU Preconditioner Done')

        # Solver tolerances
        ksp.setTolerances(rtol=1e-10, max_it=5000)

        # Iteration for Power Method
        dPHI_temp = PETSc.Vec().createSeq(self.group * max(self.conv_tri))
        dPHI_temp.set(1.0+0j)
        errdPHI = 1.0
        iter_count = 0

        while errdPHI > self.tol:
            dPHI_temp_old = dPHI_temp.copy()

            # Solve the linear system using PETSc KSP
            ksp.solve(S, dPHI_temp)

            # Calculate errors
            dPHI_diff = dPHI_temp.copy()
            dPHI_diff.axpy(-1.0, dPHI_temp_old)
            errdPHI = dPHI_diff.norm() / (dPHI_temp.norm() + 1E-20)

            iter_count += 1
            print(f'Iteration: {iter_count}, errflux = {errdPHI:.5e}')

        return dPHI_temp.getArray().copy()

class PowerMethodSolver3DRect:
    def __init__(self, group, N, conv, M, F, dx, dy, dz, precond, tol):
        self.group = group
        self.N = N
        self.M = M
        self.F = F
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.tol = tol
        self.precond = precond
        self.conv = conv

    def solve(self):
        M_petsc = PETSc.Mat().createAIJ(size=self.M.shape, csr=(self.M.indptr, self.M.indices, self.M.data))
        F_petsc = PETSc.Mat().createAIJ(size=self.F.shape, csr=(self.F.indptr, self.F.indices, self.F.data))
        M_petsc.assemble()
        F_petsc.assemble()

        # PETSc Solver (KSP) and Preconditioner (PC)
        ksp = PETSc.KSP().create()
        ksp.setOperators(M_petsc)
        ksp.setType(PETSc.KSP.Type.GMRES)

        # Preconditioner setup
        pc = ksp.getPC()
        if self.precond == 0:
            print(f'Solving using Sparse Solver')
            pc.setType(PETSc.PC.Type.NONE)
        elif self.precond == 1:
            print(f'Solving using ILU')
            pc.setType(PETSc.PC.Type.ILU)
            print(f'ILU Preconditioner Done')
        elif self.precond == 2:
            print('Solving using LU Decomposition')
            pc.setType(PETSc.PC.Type.LU)
            print(f'LU Preconditioner Done')

        # Solver tolerances
        ksp.setTolerances(rtol=1e-10, max_it=5000)

        # Iteration for Power Method
        phi_temp = PETSc.Vec().createSeq(self.group * max(self.conv))
        phi_temp.set(1.0)
        keff = 1.0
        errflux = errkeff = 1.0
        iter_count = 0

        S = phi_temp.duplicate()
        Fphi_old = phi_temp.duplicate()
        Fphi_new = phi_temp.duplicate()

        w = phi_temp.duplicate()
        vol = self.dx * self.dy * self.dz
        w.set(vol)

        while errflux > self.tol and errkeff > self.tol:
            phi_temp_old = phi_temp.copy()
            k_old = keff

            # S = 1 / k_old * (self.F @ phi_temp_old)
            F_petsc.mult(phi_temp_old, S)
            S.scale(1 / k_old)

            # Solve the linear system using PETSc KSP
            ksp.solve(S, phi_temp)

            F_petsc.mult(phi_temp_old, Fphi_old)
            F_petsc.mult(phi_temp, Fphi_new)

            # Update keff
            num = Fphi_new.dot(w)
            den = Fphi_old.dot(w)
            keff = k_old * num / den

            # Normalization
            _, max_val = phi_temp.max()
            phi_temp.scale(1.0/max_val)

            # Calculate errors
            phi_diff = phi_temp.copy()
            phi_diff.axpy(-1.0, phi_temp_old)
            errflux = phi_diff.norm() / (phi_temp.norm() + 1E-20)
            errkeff = abs((keff - k_old) / k_old)

            iter_count += 1
            print(f'Iteration: {iter_count}, keff = {keff:.5f}, errkeff = {errkeff:.6e}, '
                  f'errflux = {errflux:.5e}')

        return keff, phi_temp.getArray().copy()

class FixedSourceSolver3DRect:
    def __init__(self, group, N, conv, M, dS, PHI, dx, dy, dz, precond, tol):
        self.group = group
        self.N = N
        self.M = M
        self.dS = dS
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.tol = tol
        self.conv = conv
        self.PHI = PHI
        self.precond = precond

    def solve(self):
        M_petsc = PETSc.Mat().createAIJ(size=self.M.shape, csr=(self.M.indptr, self.M.indices, self.M.data), comm=PETSc.COMM_WORLD)
        dS_petsc = PETSc.Mat().createAIJ(size=self.dS.shape, csr=(self.dS.indptr, self.dS.indices, self.dS.data), comm=PETSc.COMM_WORLD)
        M_petsc.assemble()
        dS_petsc.assemble()

        PHI_vec = PETSc.Vec().createSeq(self.dS.shape[1])
        PHI_vec.setArray(self.PHI)

        S = PETSc.Vec().createSeq(self.dS.shape[0])
        dS_petsc.mult(PHI_vec, S)

        # PETSc Solver (KSP) and Preconditioner (PC)
        ksp = PETSc.KSP().create()
        ksp.setOperators(M_petsc)
        ksp.setType(PETSc.KSP.Type.GMRES)

        # Preconditioner setup
        pc = ksp.getPC()
        if self.precond == 0:
            print(f'Solving using Sparse Solver')
            pc.setType(PETSc.PC.Type.NONE)
        elif self.precond == 1:
            print(f'Solving using ILU')
            pc.setType(PETSc.PC.Type.ILU)
            print(f'ILU Preconditioner Done')
        elif self.precond == 2:
            print('Solving using LU Decomposition')
            pc.setType(PETSc.PC.Type.LU)
            print(f'LU Preconditioner Done')

        # Solver tolerances
        ksp.setTolerances(rtol=1e-10, max_it=5000)

        # Iteration for Power Method
        dPHI_temp = PETSc.Vec().createSeq(self.group * max(self.conv))
        dPHI_temp.set(1.0+0j)
        errdPHI = 1.0
        iter_count = 0

        while errdPHI > self.tol:
            dPHI_temp_old = dPHI_temp.copy()

            # Solve the linear system using PETSc KSP
            ksp.solve(S, dPHI_temp)

            # Calculate errors
            dPHI_diff = dPHI_temp.copy()
            dPHI_diff.axpy(-1.0, dPHI_temp_old)
            errdPHI = dPHI_diff.norm() / (dPHI_temp.norm() + 1E-20)

            iter_count += 1
            print(f'Iteration: {iter_count}, errflux = {errdPHI:.5e}')

        return dPHI_temp.getArray().copy()

class PowerMethodSolver3DHexx:
    def __init__(self, group, conv_tri, M, F, h, dz, precond, tol):
        self.group = group
        self.M = M
        self.F = F
        self.h = h
        self.dz = dz
        self.tol = tol
        self.precond = precond
        self.conv_tri = conv_tri

    def solve(self):
        M_petsc = PETSc.Mat().createAIJ(size=self.M.shape, csr=(self.M.indptr, self.M.indices, self.M.data))
        F_petsc = PETSc.Mat().createAIJ(size=self.F.shape, csr=(self.F.indptr, self.F.indices, self.F.data))
        M_petsc.assemble()
        F_petsc.assemble()

        # PETSc Solver (KSP) and Preconditioner (PC)
        ksp = PETSc.KSP().create()
        ksp.setOperators(M_petsc)
        ksp.setType(PETSc.KSP.Type.GMRES)

        # Preconditioner setup
        pc = ksp.getPC()
        if self.precond == 0:
            print(f'Solving using Sparse Solver')
            pc.setType(PETSc.PC.Type.NONE)
        elif self.precond == 1:
            print(f'Solving using ILU')
            pc.setType(PETSc.PC.Type.ILU)
            print(f'ILU Preconditioner Done')
        elif self.precond == 2:
            print('Solving using LU Decomposition')
            pc.setType(PETSc.PC.Type.LU)
            print(f'LU Preconditioner Done')

        # Solver tolerances
        ksp.setTolerances(rtol=1e-10, max_it=5000)

        # Iteration for Power Method
        phi_temp = PETSc.Vec().createSeq(self.group * max(self.conv_tri))
        phi_temp.set(1.0)
        keff = 1.0
        errflux = errkeff = 1.0
        iter_count = 0

        S = phi_temp.duplicate()
        Fphi_old = phi_temp.duplicate()
        Fphi_new = phi_temp.duplicate()

        w = phi_temp.duplicate()
        vol = self.h**2/4*np.sqrt(3)*self.dz
        w.set(vol)

        while errflux > self.tol and errkeff > self.tol:
            phi_temp_old = phi_temp.copy()
            k_old = keff

            # S = 1 / k_old * (self.F @ phi_temp_old)
            F_petsc.mult(phi_temp_old, S)
            S.scale(1 / k_old)

            # Solve the linear system using PETSc KSP
            ksp.solve(S, phi_temp)

            F_petsc.mult(phi_temp_old, Fphi_old)
            F_petsc.mult(phi_temp, Fphi_new)

            # Update keff
            num = Fphi_new.dot(w)
            den = Fphi_old.dot(w)
            keff = k_old * num / den

            # Normalization
            _, max_val = phi_temp.max()
            phi_temp.scale(1.0/max_val)

            # Calculate errors
            phi_diff = phi_temp.copy()
            phi_diff.axpy(-1.0, phi_temp_old)
            errflux = phi_diff.norm() / (phi_temp.norm() + 1E-20)
            errkeff = abs((keff - k_old) / k_old)

            iter_count += 1
            print(f'Iteration: {iter_count}, keff = {keff:.5f}, errkeff = {errkeff:.6e}, '
                  f'errflux = {errflux:.5e}')

        return keff, phi_temp.getArray().copy()

class FixedSourceSolver3DHexx:
    def __init__(self, group, conv_tri, M, dS, PHI, precond, tol):
        self.group = group
        self.M = M
        self.dS = dS
        self.tol = tol
        self.conv_tri = conv_tri
        self.PHI = PHI
        self.precond = precond

    def solve(self):
        M_petsc = PETSc.Mat().createAIJ(size=self.M.shape, csr=(self.M.indptr, self.M.indices, self.M.data), comm=PETSc.COMM_WORLD)
        dS_petsc = PETSc.Mat().createAIJ(size=self.dS.shape, csr=(self.dS.indptr, self.dS.indices, self.dS.data), comm=PETSc.COMM_WORLD)
        M_petsc.assemble()
        dS_petsc.assemble()

        PHI_vec = PETSc.Vec().createSeq(self.dS.shape[1])
        PHI_vec.setArray(self.PHI)

        S = PETSc.Vec().createSeq(self.dS.shape[0])
        dS_petsc.mult(PHI_vec, S)

        # PETSc Solver (KSP) and Preconditioner (PC)
        ksp = PETSc.KSP().create()
        ksp.setOperators(M_petsc)
        ksp.setType(PETSc.KSP.Type.GMRES)

        # Preconditioner setup
        pc = ksp.getPC()
        if self.precond == 0:
            print(f'Solving using Sparse Solver')
            pc.setType(PETSc.PC.Type.NONE)
        elif self.precond == 1:
            print(f'Solving using ILU')
            pc.setType(PETSc.PC.Type.ILU)
            print(f'ILU Preconditioner Done')
        elif self.precond == 2:
            print('Solving using LU Decomposition')
            pc.setType(PETSc.PC.Type.LU)
            print(f'LU Preconditioner Done')

        # Solver tolerances
        ksp.setTolerances(rtol=1e-10, max_it=5000)

        # Iteration for Power Method
        dPHI_temp = PETSc.Vec().createSeq(self.group * max(self.conv_tri))
        dPHI_temp.set(1.0+0j)
        errdPHI = 1.0
        iter_count = 0

        while errdPHI > self.tol:
            dPHI_temp_old = dPHI_temp.copy()

            # Solve the linear system using PETSc KSP
            ksp.solve(S, dPHI_temp)

            # Calculate errors
            dPHI_diff = dPHI_temp.copy()
            dPHI_diff.axpy(-1.0, dPHI_temp_old)
            errdPHI = dPHI_diff.norm() / (dPHI_temp.norm() + 1E-20)

            iter_count += 1
            print(f'Iteration: {iter_count}, errflux = {errdPHI:.5e}')

        return dPHI_temp.getArray().copy()
