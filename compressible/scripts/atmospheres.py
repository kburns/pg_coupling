"""
Tools for solving the structure of a stratified atmosphere with fixed gravity.

Hydrostatic equilibrium is used to write the buoyancy frequency as:
    N(z)**2 = g * ((1/γ)*dz(p)/p - dz(ρ)/ρ)
            = g * ((1/γ)*dz(p)/p - dzz(p)/dz(p))

This is used to solve for p(z) given N(z)**2, g, and γ:
    dzz(p) = ((1/γ)*dz(p)/p - N**2/g) * dz(p)

Notation:
    State vector: X = [p, pz]
    Parameters: P = [N2(z), g, γ]

"""

import numpy as np
import scipy.integrate as integ
import dedalus.public as de
import mpi4py.MPI as MPI
import logging
logger = logging.getLogger(__name__)


def deriv_scipy(X, z, *P):
    """Compute state vector derivatives for scipy ODE integration."""
    # Unpack state vector and parameters
    p, pz = X
    N2, g, γ = P
    # Compute derivatives
    dz_p = pz
    dz_pz = (pz/p/γ - N2(z)/g) * pz
    # Return derivative vector
    return np.array([dz_p, dz_pz])

def solve_scipy(X0, P, z):
    """Solve for structure using scipy ODE integration."""
    X = integ.odeint(deriv_scipy, X0, z, P)
    p = X[:,0]
    pz = X[:,1]
    return p, pz

def solve_dedalus(X0, P, domain, tolerance=1e-10, **bvp_kw):
    """Solve for structure using Dedalus NLBVP."""
    # Unpack state vector and parameters
    p0, pz0 = X0
    N2, g, γ = P
    # Setup buoyancy frequency field
    z = domain.grid(0)
    N2f = domain.new_field()
    N2f['g'] = np.array([N2(zi) for zi in z])
    # Setup NLBVP for background
    problem = de.NLBVP(domain, variables=['p','pz'], **bvp_kw)
    problem.parameters['γ'] = γ
    problem.parameters['g'] = g
    problem.parameters['p0'] = p0
    problem.parameters['pz0'] = pz0
    problem.parameters['N2'] = N2f
    problem.add_equation("dz(pz) + (N2/g)*pz = pz*pz/p/γ")
    problem.add_equation("dz(p) - pz = 0")
    problem.add_bc("left(p) = p0")
    problem.add_bc("left(pz) = pz0")
    # Start from scipy solution
    z0 = domain.bases[0].interval[0]
    zs = np.concatenate([[z0], z])
    p, pz = solve_scipy(X0, P, zs)
    solver = problem.build_solver()
    solver.state['p']['g'] = p[1:]
    solver.state['pz']['g'] = pz[1:]
    # Solve
    pert = solver.perturbations.data
    pert.fill(1+tolerance)
    while np.sum(np.abs(pert)) > tolerance:
        solver.newton_iteration()
        logger.info('Perturbation norm: {}'.format(np.sum(np.abs(pert))))
    p = solver.state['p']
    pz = solver.state['pz']
    return p, pz

def solve_hydrostatic_pressure(param, dtype):
    """Build domain and solve hydrostatic pressure from parameters."""
    # NLBVP domain
    z_basis = de.Chebyshev('z', param.z_res, interval=(0, param.Lz), dealias=2)
    domain = de.Domain([z_basis], grid_dtype=dtype, comm=MPI.COMM_SELF)
    # Solve NLBVP for background
    X0 = np.array([param.p_bottom, -param.ρ_bottom*param.g])
    P = (param.N2_func, param.g, param.γ)
    p_bvp, pz_bvp = solve_dedalus(X0, P, domain, tolerance=param.nlbvp_tolerance, ncc_cutoff=param.nlbvp_cutoff, max_ncc_terms=param.nlbvp_max_terms)
    return domain, p_bvp

def truncate_background(param, p_full):
    """Truncate background fields."""
    # Filter pressure
    p = p_full.domain.new_field()
    p['c'] = p_full['c']
    p['c'][np.abs(p['c']) < param.pressure_floor] = 0
    # Construct new density
    g, γ = param.g, param.γ
    pz = p.differentiate('z')
    a_full = ((-g)/pz).evaluate()
    # Filter density
    a = a_full.domain.new_field()
    a['c'] = a_full['c']
    a['c'][np.abs(a['c']) < param.background_floor] = 0
    # Compute hydrostatic balance
    heq = (a*pz + g).evaluate()
    # Compute buoyancy frequency
    az = a.differentiate('z')
    N2 = (g*(az/a + (1/γ)*pz/p)).evaluate()
    # Re-zero high modes
    p['c'][np.abs(p['c']) < param.pressure_floor] = 0
    a['c'][np.abs(a['c']) < param.background_floor] = 0
    return p_full, p, a_full, a, heq, N2
