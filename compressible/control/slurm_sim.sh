#!/bin/bash

# Partition             Nodes   S-C-T   Timelimit
# ---------             -----   -----   ---------
# sched_mit_hill        (32)    2-8-1   12:00:00
# sched_any_quicktest   2       2-8-1   00:15:00
# newnodes              (32)    2-10-1  12:00:00

# Job
#SBATCH --partition=sched_mit_hill
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=16
#SBATCH --time=12:00:00

# Streams
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err

# E-mail
#SBATCH --mail-user=keaton.burns+hpcjobs@gmail.com

# Content
mpirun python3 ../scripts/simulation.py
. ../control/mpi_post_checkpoints.sh
. ../control/mpi_post_scalars.sh
. ../control/mpi_post_snapshots.sh

