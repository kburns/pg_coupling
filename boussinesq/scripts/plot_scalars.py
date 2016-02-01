"""
Plot scalars from single analysis file.

Usage:
    plot_scalars.py <file> [--output=<dir>]

Options:
    --output=<output>  Output file [default: ./img_scalars.pdf]

"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()


def main(filename, output):
    """Plot scalar time-series."""

    # Data selection
    tasks = ['integ(KE_P)', 'integ(KE_N)']
    slices = (slice(None), 0, 0)

    # Plot tasks
    fig = plt.figure(figsize=(8,6))
    with h5py.File(filename, mode='r') as file:
        sim_time = file['scales']['sim_time'][:]
        for task in tasks:
            dset = file['tasks'][task]
            plt.semilogy(sim_time, dset[slices], label=task)

    # Finalize figure
    plt.title('Kinetic energy evolution')
    plt.xlabel('sim time')
    plt.legend(loc='lower right')
    plt.savefig(output)


if __name__ == "__main__":

    from docopt import docopt

    args = docopt(__doc__)
    main(args['<file>'], args['--output'])

