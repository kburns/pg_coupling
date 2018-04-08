"""
Plot scalars from single analysis file.

Usage:
    save_modes.py <krel>...

"""

import numpy as np
import modes
from mpi4py import MPI


def save_eigenmodes(output, *args, **kw):
    """Save eigenmodes."""
    evals, evecs, adj_evals, adj_evecs, proj = modes.compute_eigenmodes(*args, **kw)
    np.savez(output, evals=evals,
                     evecs=evecs,
                     adj_evals=adj_evals,
                     adj_evecs=adj_evecs,
                     proj=proj)


def main(param, krel_list, comm=MPI.COMM_WORLD):
    """Save dense eigenmodes."""
    # Sort list for parallel evaluation
    krel_list = np.sort(krel_list)
    # Cyclically distribute over processes
    for krel in krel_list[comm.rank::comm.size]:
        kx = krel * param.k_tide
        output = "data_modes_%s.npz" %krel
        save_eigenmodes(output, param, kx, sparse=False, minreal=1e-5, maxabs=1e5)


if __name__ == "__main__":
    from docopt import docopt
    import parameters as params
    args = docopt(__doc__)
    krel_list = list(map(float, args['<krel>']))
    main(params, krel_list)

