#!/bin/bash
mpirun python3 ../scripts/merge_procs.py data_snapshots --cleanup
mpirun python3 ../scripts/plot_grid.py data_snapshots/*.h5
png2mp4 'frames_grid/*' grid.mp4 30

