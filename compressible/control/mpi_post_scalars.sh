#!/bin/bash
mpirun python3 ../scripts/merge_procs.py data_scalars --cleanup
mpirun python3 ../scripts/merge_sets.py data_scalars.h5 data_scalars/*.h5
mpirun python3 ../scripts/plot_scalars.py data_scalars.h5

