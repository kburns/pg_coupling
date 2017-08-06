

import numpy as np
import dedalus.public as de
import atmospheres as atmos
import mpi4py.MPI as MPI
import logging
logger = logging.getLogger(__name__)


def linear_tide_1d(param, comm=MPI.COMM_SELF):
    # Background BVP
    domain, p_bvp = atmos.solve_hydrostatic_pressure(param, np.complex128, comm=comm)
    p_full, p0, a_full, a0, heq, N2 = atmos.truncate_background(param, p_bvp)

    # Adiabatic viscous fully-compressible hydrodynamics
    problem = de.LBVP(domain, variables=['a1','p1','u','w','uz','wz'],
        ncc_cutoff=param.ivp_cutoff, entry_cutoff=param.matrix_cutoff)
    problem.meta[:]['z']['dirichlet'] = True
    problem.parameters['a0'] = a0
    problem.parameters['p0'] = p0
    problem.parameters['a0z'] = a0.differentiate('z')
    problem.parameters['p0z'] = p0.differentiate('z')
    problem.substitutions['a0x'] = '0' #a0.differentiate('x')
    problem.substitutions['p0x'] = '0' #p0.differentiate('x')
    problem.parameters['U'] = param.U
    problem.parameters['μ'] = param.μ
    problem.parameters['γ'] = param.γ
    problem.parameters['k'] = param.k_tide
    problem.parameters['ω'] = param.ω_tide
    problem.parameters['σ'] = param.σ_tide
    problem.parameters['A'] = param.A_tide
    problem.parameters['Lz'] = param.Lz
    problem.substitutions['dt(Q)'] = "0*Q"
    problem.substitutions['dx(Q)'] = "1j*k*Q"
    problem.substitutions['ux'] = "dx(u)"
    problem.substitutions['wx'] = "dx(w)"
    problem.substitutions['div_u'] = "ux + wz"
    problem.substitutions['txx'] = "μ*(2*ux - 2/3*div_u)"
    problem.substitutions['txz'] = "μ*(wx + uz)"
    problem.substitutions['tzz'] = "μ*(2*wz - 2/3*div_u)"
    problem.substitutions['φ'] = "A*exp(k*(z - Lz))"
    problem.substitutions['cs20'] = "γ*p0*a0"
    problem.add_equation("dt(u) + U*ux + a0*dx(p1) + a1*p0x - a0*(dx(txx) + dz(txz)) = - dx(φ)")
    problem.add_equation("dt(w) + U*wx + a0*dz(p1) + a1*p0z - a0*(dx(txz) + dz(tzz)) = - dz(φ)")
    problem.add_equation("dt(a1) + U*dx(a1) + u*a0x + w*a0z -   a0*div_u = 0")
    problem.add_equation("dt(p1) + U*dx(p1) + u*p0x + w*p0z + γ*p0*div_u = 0")
    problem.add_equation("uz - dz(u) = 0")
    problem.add_equation("wz - dz(w) = 0")
    problem.add_bc("left(txz) = 0")
    problem.add_bc("right(txz) = 0")
    problem.add_bc("left(w) = 0")
    problem.add_bc("right(w) = 0")
    # Build solver
    solver = problem.build_solver()
    solver.solve()
    return domain, solver.state


def linear_tide_2d(param, comm=MPI.COMM_WORLD):
    # Solve background
    _, p_bvp = atmos.solve_hydrostatic_pressure(param, np.float64)
    p_full, p_trunc, a_full, a_trunc, heq, N2 = atmos.truncate_background(param, p_bvp)

    # IVP domain
    x_basis = de.Fourier('x', param.Nx, interval=(0, param.Lx), dealias=3/2)
    z_basis = de.Chebyshev('z', param.Nz, interval=(0, param.Lz), dealias=3/2)
    domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64, comm=comm)

    # Background state
    a0 = domain.new_field()
    p0 = domain.new_field()
    a0.meta['x']['constant'] = True
    p0.meta['x']['constant'] = True
    a0.set_scales(1)
    p0.set_scales(1)
    a_trunc.set_scales(1)
    p_trunc.set_scales(1)
    slices = domain.dist.grid_layout.slices(scales=1)
    a0['g'][:] = a_trunc['g'][slices[1]]
    p0['g'][:] = p_trunc['g'][slices[1]]

    # Adiabatic viscous fully-compressible hydrodynamics
    problem = de.LBVP(domain, variables=['a1','p1','u','w','uz','wz'],
        ncc_cutoff=param.ivp_cutoff, entry_cutoff=param.matrix_cutoff)
    problem.meta[:]['z']['dirichlet'] = True
    problem.parameters['a0'] = a0
    problem.parameters['p0'] = p0
    problem.parameters['a0z'] = a0.differentiate('z')
    problem.parameters['p0z'] = p0.differentiate('z')
    problem.substitutions['a0x'] = '0' #a0.differentiate('x')
    problem.substitutions['p0x'] = '0' #p0.differentiate('x')
    problem.parameters['U'] = param.U
    problem.parameters['μ'] = param.μ
    problem.parameters['γ'] = param.γ
    problem.parameters['k'] = param.k_tide
    problem.parameters['ω'] = param.ω_tide
    problem.parameters['σ'] = param.σ_tide
    problem.parameters['A'] = param.A_tide
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
    problem.add_bc("left(txz) = 0", condition="nx != 0")
    problem.add_bc("right(txz) = 0", condition="nx != 0")
    problem.add_bc("left(w) = 0", condition="nx != 0")
    problem.add_bc("right(w) = 0", condition="nx != 0")
    problem.add_equation("a1 = 0", condition="nx == 0")
    problem.add_equation("p1 = 0", condition="nx == 0")
    problem.add_equation("u = 0", condition="nx == 0")
    problem.add_equation("w = 0", condition="nx == 0")
    problem.add_equation("uz = 0", condition="nx == 0")
    problem.add_equation("wz = 0", condition="nx == 0")
    return domain, problem

