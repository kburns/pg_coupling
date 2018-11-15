

import numpy as np
import pathlib
import tides
import pickle
from scipy.linalg import eig
from dedalus.tools.sparse import scipy_sparse_eigs
import dedalus.public as de
import os

import logging
logger = logging.getLogger(__name__)


def filename(param, krel):
    return "emodes_mu_%.1e/emodes_%i_%.1f.pkl" %(param.μ, param.Nz, krel)


def save_modes(param, krel, verbose=True):
    """Compute and save eigenmodes."""
    os.makedirs(os.path.dirname(filename(param, krel)), exist_ok=True)
    if verbose:
        print("Saving modes krel=%.1f" %krel)
    logging.disable(logging.INFO)
    eigenmodes = compute_eigenmodes(param, krel*param.k_tide, sparse=False)
    logging.disable(logging.NOTSET)
    # Drop solver/pencil object references
    eigenmodes = eigenmodes[:5]
    pickle.dump(eigenmodes, open(filename(param, krel), "wb"))
    return eigenmodes


def load_modes(param, krel, verbose=True):
    """Load saved eigenmodes."""
    if verbose:
        print("Loading modes krel=%.1f" %krel)
    return pickle.load(open(filename(param, krel), "rb"))


def get_modes(param, krel_list, force=False, verbose=True):
    """Retrieve eigenmodes."""
    eigenmodes = {}
    for krel in krel_list:
        if force or (not pathlib.Path(filename(param, krel)).exists()):
            eigenmodes[krel] = save_modes(param, krel, verbose=verbose)
        eigenmodes[krel] = load_modes(param, krel, verbose=verbose)
    return eigenmodes


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
    Lx = solver.problem.parameters['Lx']
    ρ0 = 1 / solver.problem.parameters['a0']
    u = solver.state['u']
    w = solver.state['w']
    E_op = 4 * Lx * de.operators.integrate(ρ0*(u*np.conj(u) + w*np.conj(w)), 'z')
    # Evaluate energy for each mode
    N = len(solver.eigenvalues)
    energies = np.zeros(N)
    for i in range(N):
        solver.set_state(i)
        energies[i] = np.abs(E_op.evaluate()['c'][0])
    return energies


def compute_eigenmodes(param, kx, sparse=True, N=None, target=None, minabs=0, maxabs=np.inf, energy_norm=True):
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
        # Convert target frequency to eval
        target = -1j * target
        # Solve forward problem
        solver.solve_sparse(pencil, N=N, target=target)
        # Solve adjoint problem
        solver.adjoint_eigenvalues, solver.adjoint_eigenvectors = scipy_sparse_eigs(A=pencil.L_exp.getH(), B=-pencil.M_exp.getH(), N=N, target=np.conj(target))
    # Solve dense
    else:
        # Solve forward and adjoint problem
        solver.solve_dense(pencil, left=True, right=True, overwrite_a=True, overwrite_b=True)
        solver.adjoint_eigenvalues = solver.eigenvalues.conj()
        solver.adjoint_eigenvectors = solver.left_eigenvectors
        solver.full_eigenvalues = solver.eigenvalues.copy()
        solver.full_eigenvectors = solver.eigenvectors.copy()
        solver.full_adjoint_eigenvalues = solver.adjoint_eigenvalues.copy()
        solver.full_adjoint_eigenvectors = solver.adjoint_eigenvectors.copy()
        # Filter modes
        keep = np.isfinite(solver.eigenvalues) * (minabs < np.abs(solver.eigenvalues)) * (np.abs(solver.eigenvalues) < maxabs)
        solver.eigenvalues = solver.eigenvalues[keep]
        solver.eigenvectors = solver.eigenvectors[:,keep]
        solver.adjoint_eigenvalues = solver.adjoint_eigenvalues[keep]
        solver.adjoint_eigenvectors = solver.adjoint_eigenvectors[:,keep]
    # Sort modes
    sorting = np.argsort(1j*solver.eigenvalues)
    solver.eigenvalues = solver.eigenvalues[sorting]
    solver.eigenvectors = solver.eigenvectors[:,sorting]
    sorting = np.argsort(1j*solver.adjoint_eigenvalues.conj())
    solver.adjoint_eigenvalues = solver.adjoint_eigenvalues[sorting]
    solver.adjoint_eigenvectors = solver.adjoint_eigenvectors[:,sorting]
    # Check mode matching
    logger.info("Max eval mismatch: %e" %np.max(np.abs(solver.eigenvalues - solver.adjoint_eigenvalues.conj())))
    if not np.allclose(solver.eigenvalues, solver.adjoint_eigenvalues.conj()):
        logger.warn("WARNING: Adjoint modes may not match forward modes.")
    # Normalize modes
    # Normalize phase to that of first index
    phase = lambda Z: Z / np.abs(Z)
    solver.eigenvectors /= phase(solver.eigenvectors[0:1,:])
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
    # Check orthogonality
    metric = projector @ solver.eigenvectors
    logger.info("Max metric mismatch: %e" %np.max(np.abs(metric - np.eye(*metric.shape))))
    if not np.allclose(metric, np.eye(*metric.shape)):
        logger.warn("WARNING: Adjoint modes may not be orthogonal to forward modes.")
    return solver.eigenvalues, solver.eigenvectors, solver.adjoint_eigenvalues, solver.adjoint_eigenvectors, projector, solver, pencil

