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

