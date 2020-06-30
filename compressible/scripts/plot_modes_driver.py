

from mpi4py import MPI
import plot_modes
import modelist


rank = MPI.COMM_WORLD.rank
size = MPI.COMM_WORLD.size

filename = "data_snapshots_coeff.h5"
N = modelist.N
modes = modelist.modes
krels = modelist.krel

# Create output directory if needed
import pathlib
from dedalus.tools.parallel import Sync
output_path = pathlib.Path('img_modes')
with Sync() as sync:
    if sync.comm.rank == 0:
        if not output_path.exists():
            output_path.mkdir()

n = 0
for krel in krels:
    for family, target in modes.items():
        if (n % size) == rank:        
            output = "img_modes/img_modes_%s_%s.pdf" %(family, krel)
            print(output)
            plot_modes.main(filename, krel, N, target, output)
        n += 1


