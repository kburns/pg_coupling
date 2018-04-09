"""
Plot scalars from single analysis file.

Usage:
    plot_modes.py <file> <krel> <N> <target> [--output=<dir>]

Options:
    --output=<output>  Output file [default: ./img_modes.pdf]

"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import parameters as param
import modes


def cycle_arrays(arrays, axis):
    # Build joint shape
    n_arrays = len(arrays)
    array_shape = arrays[0].shape
    joint_shape = list(array_shape)
    joint_shape[axis] *= n_arrays
    # Build joint array
    joint_array = np.zeros(joint_shape, dtype=arrays[0].dtype)
    slices = [slice(None) for d in joint_shape]
    for i, array in enumerate(arrays):
        slices[axis] = slice(i, None, n_arrays)
        joint_array[tuple(slices)] = array
    return joint_array


def load_state_vectors(filename, kx):
    # Get state vectors
    fields = ['a1','p1','u','w','uz','wz']
    with h5py.File(filename, mode='r') as file:
        # Get kx index
        ix = list(file['scales']['kx']).index(kx)
        slices = (slice(None), ix, slice(None))
        # Gather t-z slices
        sim_time = file['scales']['sim_time'][:]
        data = [file['tasks'][f][slices] for f in fields]
        data = cycle_arrays(data, axis=-1)
    return sim_time, data


def main(filename, krel, N, target, output):

    # Parameters
    kx = krel * param.k_tide

    # Project modes
    evals, projector = modes.compute_eigenmodes(param, kx, N=N, target=target)
    sim_time, data = load_state_vectors(filename, kx)
    mode_amplitudes = data @ projector.T

    # Plot tasks
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(1, 1, 1)
    for n in range(N):
        ax.plot(sim_time, np.log10(np.abs(mode_amplitudes[:,n])), label=evals[n])
    ax.set_xlabel('sim time')
    ax.set_ylabel('mode amplitude')
    plt.legend(loc='lower right')
    plt.savefig(output)


if __name__ == "__main__":
    from docopt import docopt
    args = docopt(__doc__)
    main(args['<file>'], float(args['<krel>']), int(args['<N>']), complex(args['<target>']), args['--output'])

