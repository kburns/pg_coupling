"""
Plot mode amplitudes from coefficient snapshots.

Usage:
    plot_modes_2.py <files>... [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./frames_modes]

"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import parameters as param
plt.ioff()


def cycle_arrays(arrays, axis):
    """Create state-vector ordered array."""
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


def build_state_vector(file, index, krel):
    """Build state vector from coeff output."""
    # Get state vectors
    fields = ['a1','p1','u','w','uz','wz']
    # Get kx index
    kx = krel * param.k_tide
    ix = list(file['scales']['kx']).index(kx)
    # Gather coefficients
    slices = (index, ix, slice(None))
    data = [file['tasks'][f][slices] for f in fields]
    data = cycle_arrays(data, axis=-1)
    return data


def main(filename, start, count, output):
    """Save plot of specified tasks for given range of analysis writes."""

    # Plot settings
    krel_list = [0.25, 0.5, 1.0, 2.0, 3.0, 4.0]
    scale = 2.5
    dpi = 100
    # Layout
    nrows, ncols = 2, 3
    image = plot_tools.Box(2, 2)
    pad = plot_tools.Frame(0.2, 0.2, 0.1, 0.1)
    margin = plot_tools.Frame(0.3, 0.2, 0.1, 0.1)

    # Load projectors
    def get_proj(krel):
        with np.load("data_modes_%s.npz" %krel, mode='r') as file:
            proj = file['proj']
        return proj
    proj = {krel: get_proj(krel) for krel in krel_list}

    # Create multifigure
    mfig = plot_tools.MultiFigure(nrows, ncols, image, pad, margin, scale)
    fig = mfig.figure

    # Plot writes
    with h5py.File(filename, mode='r') as file:
        for index in range(start, start+count):
            for n, krel in enumerate(krel_list):
                # Build subfigure axes
                i, j = divmod(n, ncols)
                ax = mfig.add_axes(i, j, [0, 0, 1, 1])
                # Plot amplitudes
                data = build_state_vector(file, index, krel)
                amps = proj[krel] @ data
                ax.loglog(evals.real, np.abs(amps), 'ob')
                ax.loglog(-evals.real, np.abs(amps), '.r')
                ax.set_title('krel = %s' %krel)
                ax.grid()
                ax.set_xlabel('ω')
                ax.ylabel('amp')
            # Add time title
            title = title_func(file['scales/sim_time'][index])
            title_height = 1 - 0.5 * mfig.margin.top / mfig.fig.y
            fig.suptitle(title, x=0.48, y=title_height, ha='left')
            # Save figure
            savename = savename_func(file['scales/write_number'][index])
            savepath = output.joinpath(savename)
            fig.savefig(str(savepath), dpi=dpi)
            fig.clear()
    plt.close(fig)


if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus.tools import logging
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync

    # Parse arguments
    args = docopt(__doc__)
    output_path = pathlib.Path(args['--output']).absolute()

    # Create output directory if needed
    with Sync() as sync:
        if sync.comm.rank == 0:
            if not output_path.exists():
                output_path.mkdir()

    # Visit writes with main function
    post.visit_writes(args['<files>'], main, output=output_path)

