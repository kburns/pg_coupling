#!/bin/bash
mpirun python3 ../scripts/merge_procs.py data_snapshots --cleanup
mpirun python3 ../scripts/merge_procs.py data_snapshots_coeff --cleanup
mpirun python3 ../scripts/merge_sets.py data_snapshots_coeff.h5 data_snapshots_coeff/*.h5
mpirun python3 ../scripts/plot_grid.py data_snapshots/*.h5
mpirun python3 ../scripts/plot_coeffs.py data_snapshots_coeff.h5

