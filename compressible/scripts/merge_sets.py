"""
Merge analysis sets from a FileHandler.

Usage:
    merge_sets.py <joint_path> <set_paths>...

"""

if __name__ == "__main__":

    from docopt import docopt
    from mpi4py import MPI
    from dedalus.tools import logging
    from dedalus.tools import post

    args = docopt(__doc__)
    if MPI.COMM_WORLD.rank == 0:
        post.merge_sets(args['<joint_path>'], args['<set_paths>'])

