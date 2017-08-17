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
    tasks = ['KE']
    slices = (slice(None), 0, 0)

    # Plot tasks
    fig, axes = plt.subplots(len(tasks), 1, figsize=(10,8))
    if len(tasks) == 1:
        axes = [axes]

    with h5py.File(filename, mode='r') as file:
        sim_time = file['scales']['sim_time'][:]
        for task, ax in zip(tasks, axes):
            dset = file['tasks'][task]
            ax.plot(sim_time[slices[0]], dset[slices], '.-')
            ax.set_ylabel(task)
        ax.set_xlabel('sim time')

    # Finalize figure
    # plt.xlabel('sim time')
    # plt.legend(loc='lower right')
    plt.savefig(output)

if __name__ == "__main__":
    from docopt import docopt
    args = docopt(__doc__)
    main(args['<file>'], args['--output'])

