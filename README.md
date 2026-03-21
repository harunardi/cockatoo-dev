# COCKATOO
## A Forward, Adjoint, and Neutron Noise General Tool

**Documentation available at**: [coming soon]

**Features:**
* Performs forward, adjoint, and neutron noise simulations.
* Solves multi-group of neutron energy.
* Finite volume for spatial discretization.
* Diffusion approximation for angular discretization.
* Uses PETSc for linear solver.
* Supports LU and ILU preconditioning.
* Applicable to various geometries: 1D, 2D rectangular, 2D hexagonal, 3D rectangular, 3D hexagonal
* Performs neutron noise unfolding methods with various methods (inversion, zoning, scanning, brute force, backward elimination, greedy)
* All python based

# COCKATOO
Cockatoo is a forward, adjoint, and neutron noise general tool that solves neutron diffusion equation. Cockatoo uses neutron noise equation in frequency domain formulation. It is the continuation of [FANGS-UNFOLD](https://github.com/harunardi/FANGS-UNFOLD), author's tool developed for his dissertation. 

# User Guides

[coming soon]

# How to cite
[coming soon]

# NOTE: Required libraries to run the codes
The required dependencies are as follows:

| Package  | Minimum Version |
| ------------- | ------------- |
| numpy  | 2.4  |
| scipy  | 1.17  |
| matplotlib  | 3.10  |
| shapely  | 2.1  |
| h5py  | 3.15  |
| petsc4py  | 3.24  |
| slepc4py  | 3.24  |

It is recommended that the code is installed in conda environment. The command line is as follows:

    conda create --name noise numpy scipy matplotlib shapely h5py petsc4py slepc4py -c conda-forge
    conda install -c conda-forge 'petsc=*=complex*' petsc4py
