

import numpy as np
import dedalus.public as de
import background as background
import mpi4py.MPI as MPI
import logging
from collections import OrderedDict
logger = logging.getLogger(__name__)


class Atmosphere:
    """Class representing combined domain and background."""

    def __init__(self, param, dim, dtype=np.float64, comm=None):
        # Solve and truncate background
        bvp_domain, p0_full, a0_full = background.solve_hydrostatic_pressure(param, dtype)
        p0_full, p0_trunc, a0_full, a0_trunc, heq, N2 = background.truncate_background(param, p0_full, a0_full)
        # Save domain and backgrounds
        if dim == 1:
            # Use BVP results
            self.domain = bvp_domain
            self.a0 = a0_trunc
            self.p0 = p0_trunc
        elif dim == 2:
            # Construct new domain
            x_basis = de.Fourier('x', param.Nx, interval=(0, param.Lx), dealias=2)
            z_basis = de.Chebyshev('z', param.Nz, interval=(0, param.Lz), dealias=2)
            self.domain = de.Domain([x_basis, z_basis], grid_dtype=dtype, comm=comm)
            # Background state
            self.a0 = a0 = self.domain.new_field()
            self.p0 = p0 = self.domain.new_field()
            a0.meta['x']['constant'] = True
            p0.meta['x']['constant'] = True
            a0.set_scales(1)
            p0.set_scales(1)
            a0_trunc.set_scales(1)
            p0_trunc.set_scales(1)
            slices = self.domain.dist.grid_layout.slices(scales=1)
            a0['g'][:] = a0_trunc['g'][slices[1]]
            p0['g'][:] = p0_trunc['g'][slices[1]]
        else:
            raise ValueError("Dimension must be 1 or 2")
        # Store derivatives
        self.a0z = self.a0.differentiate('z')
        self.p0z = self.p0.differentiate('z')


def ivp(param, atmos=None, dtype=np.float64, comm=MPI.COMM_WORLD):
    if atmos is None:
        atmos = Atmosphere(param, dim=2, dtype=dtype, comm=comm)
    # Adiabatic viscous fully-compressible hydrodynamics
    problem = de.IVP(atmos.domain, variables=['a1','p1','u','w','uz','wz'],
        ncc_cutoff=param.ivp_cutoff, entry_cutoff=param.matrix_cutoff)
    problem.meta[:]['z']['dirichlet'] = True
    problem.parameters['a0'] = atmos.a0
    problem.parameters['p0'] = atmos.p0
    problem.parameters['a0z'] = atmos.a0z
    problem.parameters['p0z'] = atmos.p0z
    problem.parameters['U'] = param.U
    problem.parameters['μ'] = param.μ
    problem.parameters['γ'] = param.γ
    problem.parameters['k'] = param.k_tide
    problem.parameters['ω'] = param.ω_tide
    problem.parameters['σ'] = param.σ_tide
    problem.parameters['A'] = param.A_tide
    problem.parameters['Lz'] = param.Lz
    problem.substitutions['a0x'] = '0'
    problem.substitutions['p0x'] = '0'
    problem.substitutions['ux'] = "dx(u)"
    problem.substitutions['wx'] = "dx(w)"
    problem.substitutions['div_u'] = "ux + wz"
    problem.substitutions['txx'] = "μ*(2*ux - 2/3*div_u)"
    problem.substitutions['txz'] = "μ*(wx + uz)"
    problem.substitutions['tzz'] = "μ*(2*wz - 2/3*div_u)"
    problem.substitutions['φ'] = "A*exp(σ*t)*cos(k*x)*exp(k*(z - Lz))"
    problem.substitutions['cs20'] = "γ*p0*a0"
    problem.add_equation("dt(u) + U*ux + a0*dx(p1) + a1*p0x - a0*(dx(txx) + dz(txz)) = - (u*ux + w*uz) - a1*dx(p1) + a1*(dx(txx) + dz(txz)) - dx(φ)")
    problem.add_equation("dt(w) + U*wx + a0*dz(p1) + a1*p0z - a0*(dx(txz) + dz(tzz)) = - (u*wx + w*wz) - a1*dz(p1) + a1*(dx(txz) + dz(tzz)) - dz(φ)")
    problem.add_equation("dt(a1) + U*dx(a1) + u*a0x + w*a0z -   a0*div_u = - (u*dx(a1) + w*dz(a1)) +   a1*div_u")
    problem.add_equation("dt(p1) + U*dx(p1) + u*p0x + w*p0z + γ*p0*div_u = - (u*dx(p1) + w*dz(p1)) - γ*p1*div_u")
    problem.add_equation("uz - dz(u) = 0")
    problem.add_equation("wz - dz(w) = 0")
    problem.add_bc("left(txz) = 0")
    problem.add_bc("right(txz) = 0")
    problem.add_bc("left(w) = 0")
    problem.add_bc("right(w) = 0")
    return atmos, problem

def linear_tide_1d(param, comm=MPI.COMM_SELF):
    """Solve linear tide with k=k_tide in 1D."""
    # Background BVP
    dtype = np.complex128
    domain, p_full, a_full = atmos.solve_hydrostatic_pressure(param, dtype, comm=comm)
    p_full, p_trunc, a_full, a_trunc, heq, N2 = atmos.truncate_background(param, p_full, a_full)
    # Adiabatic viscous fully-compressible hydrodynamics
    problem = de.LBVP(domain, variables=['a1','p1','u','w','uz','wz'],
        ncc_cutoff=param.ivp_cutoff, entry_cutoff=param.matrix_cutoff)
    problem.meta[:]['z']['dirichlet'] = True
    problem.parameters['a0'] = a_trunc
    problem.parameters['p0'] = p_trunc
    problem.parameters['a0z'] = a_trunc.differentiate('z')
    problem.parameters['p0z'] = p_trunc.differentiate('z')
    problem.parameters['U'] = param.U
    problem.parameters['μ'] = param.μ
    problem.parameters['γ'] = param.γ
    problem.parameters['k'] = param.k_tide
    problem.parameters['ω'] = param.ω_tide
    problem.parameters['σ'] = param.σ_tide
    problem.parameters['A'] = param.A_tide
    problem.parameters['Lx'] = param.Lx
    problem.parameters['Lz'] = param.Lz
    problem.substitutions['dt(Q)'] = "0*Q"
    problem.substitutions['dx(Q)'] = "1j*k*Q"
    problem.substitutions['ux'] = "dx(u)"
    problem.substitutions['wx'] = "dx(w)"
    problem.substitutions['div_u'] = "ux + wz"
    problem.substitutions['txx'] = "μ*(2*ux - 2/3*div_u)"
    problem.substitutions['txz'] = "μ*(wx + uz)"
    problem.substitutions['tzz'] = "μ*(2*wz - 2/3*div_u)"
    problem.substitutions['φ'] = "A/2*exp(k*(z - Lz))"
    problem.substitutions['cs20'] = "γ*p0*a0"
    problem.add_equation("dt(u) + U*ux + a0*dx(p1) + a1*p0x - a0*(dx(txx) + dz(txz)) = - dx(φ)")
    problem.add_equation("dt(w) + U*wx + a0*dz(p1) + a1*p0z - a0*(dx(txz) + dz(tzz)) = - dz(φ)")
    problem.add_equation("dt(a1) + U*dx(a1) + u*a0x + w*a0z -   a0*div_u = 0")
    problem.add_equation("dt(p1) + U*dx(p1) + u*p0x + w*p0z + γ*p0*div_u = 0")
    problem.add_equation("uz - dz(u) = 0")
    problem.add_equation("wz - dz(w) = 0")
    problem.add_bc("left(txz/μ) = 0")
    problem.add_bc("right(txz/μ) = 0")
    problem.add_bc("left(w) = 0")
    problem.add_bc("right(w) = 0")
    return domain, problem


def linear_tide_2d(param, atmos=None, dtype=np.float64, comm=MPI.COMM_WORLD):
    """Solve linear tide in 2D."""
    if atmos is None:
        atmos = Atmosphere(param, dim=2, dtype=dtype, comm=comm)
    # Adiabatic viscous fully-compressible hydrodynamics
    problem = de.LBVP(atmos.domain, variables=['a1','p1','u','w','uz','wz'],
        ncc_cutoff=param.ivp_cutoff, entry_cutoff=param.matrix_cutoff)
    problem.meta[:]['z']['dirichlet'] = True
    problem.parameters['a0'] = atmos.a0
    problem.parameters['p0'] = atmos.p0
    problem.parameters['a0z'] = atmos.a0z
    problem.parameters['p0z'] = atmos.p0z
    problem.substitutions['a0x'] = '0'
    problem.substitutions['p0x'] = '0'
    problem.parameters['U'] = param.U
    problem.parameters['μ'] = param.μ
    problem.parameters['γ'] = param.γ
    problem.parameters['k'] = param.k_tide
    problem.parameters['ω'] = param.ω_tide
    problem.parameters['σ'] = param.σ_tide
    problem.parameters['A'] = param.A_tide
    problem.parameters['Lx'] = param.Lx
    problem.parameters['Lz'] = param.Lz
    problem.substitutions['dt(Q)'] = "0*Q"
    problem.substitutions['ux'] = "dx(u)"
    problem.substitutions['wx'] = "dx(w)"
    problem.substitutions['div_u'] = "ux + wz"
    problem.substitutions['txx'] = "μ*(2*ux - 2/3*div_u)"
    problem.substitutions['txz'] = "μ*(wx + uz)"
    problem.substitutions['tzz'] = "μ*(2*wz - 2/3*div_u)"
    problem.substitutions['φ'] = "A*cos(k*x)*exp(k*(z - Lz))"
    problem.substitutions['cs20'] = "γ*p0*a0"
    problem.add_equation("dt(u) + U*ux + a0*dx(p1) + a1*p0x - a0*(dx(txx) + dz(txz)) = - dx(φ)", condition="nx != 0")
    problem.add_equation("dt(w) + U*wx + a0*dz(p1) + a1*p0z - a0*(dx(txz) + dz(tzz)) = - dz(φ)", condition="nx != 0")
    problem.add_equation("dt(a1) + U*dx(a1) + u*a0x + w*a0z -   a0*div_u = 0", condition="nx != 0")
    problem.add_equation("dt(p1) + U*dx(p1) + u*p0x + w*p0z + γ*p0*div_u = 0", condition="nx != 0")
    problem.add_equation("uz - dz(u) = 0", condition="nx != 0")
    problem.add_equation("wz - dz(w) = 0", condition="nx != 0")
    problem.add_bc("left(txz/μ) = 0", condition="nx != 0")
    problem.add_bc("right(txz/μ) = 0", condition="nx != 0")
    problem.add_bc("left(w) = 0", condition="nx != 0")
    problem.add_bc("right(w) = 0", condition="nx != 0")
    # kx = 0 equations
    problem.add_equation("u = 0", condition="nx == 0")
    problem.add_equation("w = 0", condition="nx == 0")
    problem.add_equation("a1 = 0", condition="nx == 0")
    problem.add_equation("p1 = 0", condition="nx == 0")
    problem.add_equation("uz = 0", condition="nx == 0")
    problem.add_equation("wz = 0", condition="nx == 0")
    return atmos, problem


def eigenmodes_1d(param, kx, comm=MPI.COMM_SELF):
    """Solve normal modes for any kx in 1D."""
    # Background BVP
    domain, p_full, a_full = background.solve_hydrostatic_pressure(param, np.complex128, comm=comm)
    p_full, p_trunc, a_full, a_trunc, heq, N2 = background.truncate_background(param, p_full, a_full)
    # Adiabatic viscous fully-compressible hydrodynamics
    problem = de.EVP(domain, variables=['a1','p1','u','w','uz','wz'], eigenvalue='σ',
        ncc_cutoff=param.ivp_cutoff, entry_cutoff=param.matrix_cutoff)
    problem.meta[:]['z']['dirichlet'] = True
    problem.parameters['a0'] = a_trunc
    problem.parameters['p0'] = p_trunc
    problem.parameters['a0z'] = a_trunc.differentiate('z')
    problem.parameters['p0z'] = p_trunc.differentiate('z')
    problem.substitutions['a0x'] = '0'
    problem.substitutions['p0x'] = '0'
    problem.parameters['U'] = 0  # Look at modes in local frame
    problem.parameters['μ'] = param.μ
    problem.parameters['γ'] = param.γ
    problem.parameters['kx'] = kx  # Instead of k_tide
    problem.parameters['Lx'] = param.Lx
    problem.parameters['Lz'] = param.Lz
    problem.substitutions['dt(A)'] = "σ*A"
    problem.substitutions['dx(A)'] = "1j*kx*A"
    problem.substitutions['ux'] = "dx(u)"
    problem.substitutions['wx'] = "dx(w)"
    problem.substitutions['div_u'] = "ux + wz"
    problem.substitutions['txx'] = "μ*(2*ux - 2/3*div_u)"
    problem.substitutions['txz'] = "μ*(wx + uz)"
    problem.substitutions['tzz'] = "μ*(2*wz - 2/3*div_u)"
    problem.substitutions['cs20'] = "γ*p0*a0"
    problem.add_equation("dt(u) + U*ux + a0*dx(p1) + a1*p0x - a0*(dx(txx) + dz(txz)) = - (u*ux + w*uz) - a1*dx(p1) + a1*(dx(txx) + dz(txz))")
    problem.add_equation("dt(w) + U*wx + a0*dz(p1) + a1*p0z - a0*(dx(txz) + dz(tzz)) = - (u*wx + w*wz) - a1*dz(p1) + a1*(dx(txz) + dz(tzz))")
    problem.add_equation("dt(a1) + U*dx(a1) + u*a0x + w*a0z -   a0*div_u = - (U*a0x + u*dx(a1) + w*dz(a1)) +   a1*div_u")
    problem.add_equation("dt(p1) + U*dx(p1) + u*p0x + w*p0z + γ*p0*div_u = - (U*p0x + u*dx(p1) + w*dz(p1)) - γ*p1*div_u")
    problem.add_equation("uz - dz(u) = 0")
    problem.add_equation("wz - dz(w) = 0")
    problem.add_bc("left(txz/μ) = 0")
    problem.add_bc("right(txz/μ) = 0")
    problem.add_bc("left(w) = 0")
    problem.add_bc("right(w) = 0")
    return domain, problem

def eigenmodes_inviscid_1d(param, kx, comm=MPI.COMM_SELF):
    """Solve normal modes for any kx in 1D."""
    # Background BVP
    domain, p_full, a_full = background.solve_hydrostatic_pressure(param, np.complex128, comm=comm)
    p_full, p_trunc, a_full, a_trunc, heq, N2 = atmos.truncate_background(param, p_full, a_full)
    # Adiabatic viscous fully-compressible hydrodynamics
    problem = de.EVP(domain, variables=['a1','p1','u','w','uz','wz'], eigenvalue='σ',
        ncc_cutoff=param.ivp_cutoff, entry_cutoff=param.matrix_cutoff)
    problem.meta[:]['z']['dirichlet'] = True
    problem.parameters['a0'] = a_trunc
    problem.parameters['p0'] = p_trunc
    problem.parameters['a0z'] = a_trunc.differentiate('z')
    problem.parameters['p0z'] = p_trunc.differentiate('z')
    problem.substitutions['a0x'] = '0'
    problem.substitutions['p0x'] = '0'
    problem.parameters['U'] = 0  # Look at modes in local frame
    problem.substitutions['μ'] = '0'  # Solve for inviscid modes
    problem.parameters['γ'] = param.γ
    problem.parameters['kx'] = kx  # Instead of k_tide
    problem.parameters['Lx'] = param.Lx
    problem.parameters['Lz'] = param.Lz
    problem.substitutions['dt(A)'] = "σ*A"
    problem.substitutions['dx(A)'] = "1j*kx*A"
    problem.substitutions['ux'] = "dx(u)"
    problem.substitutions['wx'] = "dx(w)"
    problem.substitutions['div_u'] = "ux + wz"
    problem.substitutions['txx'] = "μ*(2*ux - 2/3*div_u)"
    problem.substitutions['txz'] = "μ*(wx + uz)"
    problem.substitutions['tzz'] = "μ*(2*wz - 2/3*div_u)"
    problem.substitutions['φ'] = "0"
    problem.substitutions['cs20'] = "γ*p0*a0"
    problem.add_equation("dt(u) + U*ux + a0*dx(p1) + a1*p0x - a0*(dx(txx) + dz(txz)) = - (u*ux + w*uz) - a1*dx(p1) + a1*(dx(txx) + dz(txz)) - dx(φ)")
    problem.add_equation("dt(w) + U*wx + a0*dz(p1) + a1*p0z - a0*(dx(txz) + dz(tzz)) = - (u*wx + w*wz) - a1*dz(p1) + a1*(dx(txz) + dz(tzz)) - dz(φ)")
    problem.add_equation("dt(a1) + U*dx(a1) + u*a0x + w*a0z -   a0*div_u = - (U*a0x + u*dx(a1) + w*dz(a1)) +   a1*div_u")
    problem.add_equation("dt(p1) + U*dx(p1) + u*p0x + w*p0z + γ*p0*div_u = - (U*p0x + u*dx(p1) + w*dz(p1)) - γ*p1*div_u")
    problem.add_equation("uz - dz(u) = 0")
    problem.add_equation("wz - dz(w) = 0")
    #problem.add_bc("left(txz) = 0")
    #problem.add_bc("right(txz) = 0")
    problem.add_bc("left(uz - dz(u)) = 0")
    problem.add_bc("left(w) = 0")
    problem.add_bc("right(w) = 0")
    return domain, problem

def eigenmodes_inviscid_1d_simplified(param, kx, comm=MPI.COMM_SELF):
    """Solve normal modes for any kx in 1D."""
    # Background BVP
    domain, p_full, a_full = atmos.solve_hydrostatic_pressure(param, np.complex128, comm=comm)
    p_full, p_trunc, a_full, a_trunc, heq, N2 = atmos.truncate_background(param, p_full, a_full)
    # Adiabatic viscous fully-compressible hydrodynamics
    problem = de.EVP(domain, variables=['u','w','uz','wz'], eigenvalue='σ2',
        ncc_cutoff=param.ivp_cutoff, entry_cutoff=param.matrix_cutoff)
    problem.meta[:]['z']['dirichlet'] = True
    problem.parameters['a0'] = a_trunc
    problem.parameters['p0'] = p_trunc
    problem.parameters['a0z'] = a_trunc.differentiate('z')
    problem.parameters['p0z'] = p_trunc.differentiate('z')
    problem.substitutions['a0x'] = '0'
    problem.substitutions['p0x'] = '0'
    problem.parameters['g'] = 1
    problem.parameters['U'] = 0  # Look at modes in local frame
    problem.substitutions['μ'] = '0'  # Solve for inviscid modes
    problem.parameters['γ'] = param.γ
    problem.parameters['kx'] = kx  # Instead of k_tide
    problem.parameters['Lx'] = param.Lx
    problem.parameters['Lz'] = param.Lz
    problem.substitutions['dtt(A)'] = "σ2*A"
    problem.substitutions['dx(A)'] = "1j*kx*A"
    problem.substitutions['ux'] = "dx(u)"
    problem.substitutions['wx'] = "dx(w)"
    problem.substitutions['div_u'] = "ux + wz"
    problem.substitutions['txx'] = "μ*(2*ux - 2/3*div_u)"
    problem.substitutions['txz'] = "μ*(wx + uz)"
    problem.substitutions['tzz'] = "μ*(2*wz - 2/3*div_u)"
    problem.substitutions['φ'] = "0"
    problem.substitutions['cs20'] = "γ*p0*a0"
    problem.add_equation("dtt(u) - dx(-w*g) - cs20*dx(div_u) = 0")
    problem.add_equation("dtt(w) - dz(-w*g) - cs20*dz(div_u) + (γ-1)*div_u*g = 0")
    problem.add_equation("uz - dz(u) = 0")
    problem.add_equation("wz - dz(w) = 0")
    problem.add_bc("left(uz - dz(u)) = 0")
    problem.add_bc("left(w) = 0")
    problem.add_bc("right(w) = 0")
    return domain, problem
