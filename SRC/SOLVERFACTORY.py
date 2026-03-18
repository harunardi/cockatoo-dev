from SRC.METHODS import PowerMethodSolver1D
from SRC.METHODS import FixedSourceSolver1D
from SRC.METHODS import PowerMethodSolver2DRect
from SRC.METHODS import FixedSourceSolver2DRect
from SRC.METHODS import PowerMethodSolver2DHexx
from SRC.METHODS import FixedSourceSolver2DHexx
from SRC.METHODS import PowerMethodSolver3DRect
from SRC.METHODS import FixedSourceSolver3DRect
from SRC.METHODS import PowerMethodSolver3DHexx
from SRC.METHODS import FixedSourceSolver3DHexx

class SolverFactory:
    @staticmethod
    def get_solver_power1D(solver_type, group, N, M, F, x, precond, tol):
        if solver_type in ['forward', 'adjoint']:
            return PowerMethodSolver1D(group, N, M, F, x, precond, tol)
        else:
            raise ValueError(f"Unknown solver type: {solver_type}")

    @staticmethod
    def get_solver_fixed1D(solver_type, group, N, M, dS, dSOURCE, PHI, precond, tol):
        if solver_type == 'noise':
            return FixedSourceSolver1D(group, N, M, dS, dSOURCE, PHI, precond, tol)
        else:
            raise ValueError(f"Unknown solver type: {solver_type}")    

    @staticmethod
    def get_solver_power2DRect(solver_type, group, N, conv, M, F, dx, dy, precond, tol):
        if solver_type in ['forward', 'adjoint']:
            return PowerMethodSolver2DRect(group, N, conv, M, F, dx, dy, precond, tol)
        else:
            raise ValueError(f"Unknown solver type: {solver_type}")
    
    @staticmethod
    def get_solver_fixed2DRect(solver_type, group, N, conv, M, dS, PHI, dx, dy, precond, tol):
        if solver_type == 'noise':
            return FixedSourceSolver2DRect(group, N, conv, M, dS, PHI, dx, dy, precond, tol)
        else:
            raise ValueError(f"Unknown solver type: {solver_type}")    
        
    @staticmethod
    def get_solver_power2DHexx(solver_type, group, conv_tri, M, F, h, precond, tol):
        if solver_type in ['forward', 'adjoint']:
            return PowerMethodSolver2DHexx(group, conv_tri, M, F, h, precond, tol)
        else:
            raise ValueError(f"Unknown solver type: {solver_type}")
    
    @staticmethod
    def get_solver_fixed2DHexx(solver_type, group, conv_tri, M, dS, PHI, precond, tol):
        if solver_type == 'noise':
            return FixedSourceSolver2DHexx(group, conv_tri, M, dS, PHI, precond, tol)
        else:
            raise ValueError(f"Unknown solver type: {solver_type}")    

    @staticmethod
    def get_solver_power3DRect(solver_type, group, N, conv, M, F, dx, dy, dz, precond, tol):
        if solver_type in ['forward', 'adjoint']:
            return PowerMethodSolver3DRect(group, N, conv, M, F, dx, dy, dz, precond, tol)
        else:
            raise ValueError(f"Unknown solver type: {solver_type}")

    @staticmethod
    def get_solver_fixed3DRect(solver_type, group, N, conv, M, dS, PHI, dx, dy, dz, precond, tol):
        if solver_type == 'noise':
            return FixedSourceSolver3DRect(group, N, conv, M, dS, PHI, dx, dy, dz, precond, tol)
        else:
            raise ValueError(f"Unknown solver type: {solver_type}")    

    @staticmethod
    def get_solver_power3DHexx(solver_type, group, conv_tri, M, F, h, dz, precond, tol):
        if solver_type in ['forward', 'adjoint']:
            return PowerMethodSolver3DHexx(group, conv_tri, M, F, h, dz, precond, tol)
        else:
            raise ValueError(f"Unknown solver type: {solver_type}")

    @staticmethod
    def get_solver_fixed3DHexx(solver_type, group, conv_tri, M, dS, PHI, precond, tol):
        if solver_type == 'noise':
            return FixedSourceSolver3DHexx(group, conv_tri, M, dS, PHI, precond, tol)
        else:
            raise ValueError(f"Unknown solver type: {solver_type}")    
