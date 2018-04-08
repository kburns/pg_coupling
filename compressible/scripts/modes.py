

import numpy as np
import tides
from scipy.linalg import eig
from dedalus.tools.sparse import scipy_sparse_eigs
import dedalus.public as de

import logging
logger = logging.getLogger(__name__)


def compute_energies(solver):
    """
    Compute eigenmode energies.

    U = u exp(ikx) + CC
    U^2 = u^2 exp(2ikx) + u u* + CC
    int U^2 dx = 2 Lx u u*

    E = 2 int ρ0 U.U dx dz
      = 4 Lx int ρ0 (u u* + w w*) dz
    """
    # Construct energy operator
    Lx = 2*np.pi / solver.problem.parameters['kx']
    ρ0 = 1 / solver.problem.parameters['a0']
    u = solver.state['u']
    w = solver.state['w']
    E_op = 4 * Lx * de.operators.integrate(ρ0*(u*np.conj(u) + w*np.conj(w)), 'z')
    # Evaluate energy for each mode
    N = len(solver.eigenvalues)
    energies = np.zeros(N)
    for i in range(N):
        solver.set_state(i)
        energies[i] = E_op.evaluate()['c'][0]
    return energies


def compute_eigenmodes(param, kx, sparse=True, N=None, target=None, minreal=0, maxabs=np.inf, energy_norm=True):
    """
    Solve eigenvalue problem for 1D eigenmodes and adjoint eigenmodes.
    """
    # Create eigenvalue solver
    domain, problem = tides.eigenmodes_1d(param, kx=kx)
    solver = problem.build_solver()
    pencil = solver.pencils[0]
    # Solve sparse
    if sparse:
        from dedalus.tools.sparse import scipy_sparse_eigs
        # Solve forward problem
        solver.solve_sparse(pencil, N=N, target=target)
        # Solve adjoint problem
        solver.adjoint_eigenvalues, solver.adjoint_eigenvectors = scipy_sparse_eigs(A=pencil.L_exp.getH(), B=-pencil.M_exp.getH(), N=N, target=np.conj(target))
    # Solve dense
    else:
        # Solve forward problem
        solver.solve_dense(pencil)
        # Solve adjoint problem
        solver.adjoint_eigenvalues, solver.adjoint_eigenvectors = eig(pencil.L.getH().A, b=-pencil.M.getH().A)
        # Filter modes
        keep = (np.abs(solver.eigenvalues) < maxabs) * (np.abs(solver.eigenvalues.real) > minreal)
        solver.eigenvalues = solver.eigenvalues[keep]
        solver.eigenvectors = solver.eigenvectors[:,keep]
        keep = (np.abs(solver.adjoint_eigenvalues) < maxabs) * (np.abs(solver.adjoint_eigenvalues.real) > minreal)
        solver.adjoint_eigenvalues = solver.adjoint_eigenvalues[keep]
        solver.adjoint_eigenvectors = solver.adjoint_eigenvectors[:,keep]
    # Sort modes
    sorting = np.argsort(solver.eigenvalues)
    solver.eigenvalues = solver.eigenvalues[sorting]
    solver.eigenvectors = solver.eigenvectors[:,sorting]
    sorting = np.argsort(solver.adjoint_eigenvalues.conj())
    solver.adjoint_eigenvalues = solver.adjoint_eigenvalues[sorting]
    solver.adjoint_eigenvectors = solver.adjoint_eigenvectors[:,sorting]
    # Check mode matching
    logger.info("Max eval mismatch: %e" %np.max(np.abs(solver.eigenvalues - solver.adjoint_eigenvalues.conj())))
    if not np.allclose(solver.eigenvalues, solver.adjoint_eigenvalues.conj()):
        logger.warn("WARNING: Adjoint modes may not match forward modes.")
    # Normalize modes
    if energy_norm:
        # Normalize by energy
        metric_diag = compute_energies(solver)
        solver.eigenvectors /= np.sqrt(metric_diag)
    else:
        # Normalize by Chebyshev inner product
        metric = solver.eigenvectors.T.conj() @ solver.eigenvectors
        solver.eigenvectors /= np.sqrt(np.diag(metric))
    # Normalize adjoint modes
    metric = solver.adjoint_eigenvectors.T.conj() @ pencil.M @ solver.eigenvectors
    solver.adjoint_eigenvectors /= np.diag(metric).conj()
    projector = solver.adjoint_eigenvectors.T.conj() @ pencil.M
    return solver.eigenvalues, solver.eigenvectors, solver.adjoint_eigenvalues, solver.adjoint_eigenvectors, projector

