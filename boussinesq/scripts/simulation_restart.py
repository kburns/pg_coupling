"""
2D Boussinesq hydro script for examining IGW coupling.
"""

import numpy as np
from mpi4py import MPI
import time

from dedalus import public as de
from dedalus.extras import flow_tools

from convolve import Convolve
de.operators.parseables['Convolve'] = Convolve

import logging
logger = logging.getLogger(__name__)


# Parameters
Lx, Lz = (2*np.pi, 2*np.pi) # Domain size
Rx, Rz = (64, 64)           # Resolution
N = 1                       # Buoyancy frequency
ν = 2e-4                    # Momentum diffusivity
κ = 2e-4                    # Buoyancy diffusivity
parent_nx = 1               # Parent x wave-index
parent_nz = 1               # Parent z wave-index
parent_amp = 1e-1           # Parent wave amplitude
noise_amp = 1e-4            # Noise amplitude
noise_cut = 8              # Noise cutoff wave-index

# Parent wave details
N = 1
kx = parent_nx * (2*np.pi / Lx)
kz = parent_nz * (2*np.pi / Lz)
K = (kx**2 + kz**2)**(1/2)
ω = abs(N * kx / K)

# Bases and domain
x_basis = de.Fourier('x', Rx, interval=(0, Lx), dealias=3/2)
z_basis = de.Fourier('z', Rz, interval=(0, Lz), dealias=3/2)
domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)

# Filters for isolating and excluding parent mode
cslices = domain.dist.coeff_layout.slices(scales=1)
global_kx = x_basis.wavenumbers[:, None]
global_kz = z_basis.wavenumbers[None, :]
FP = domain.new_field()
FN = domain.new_field()
FP['c'] = ((global_kx == kx) * (global_kz == kz))[cslices]
FN['c'] = ((global_kx != kx) * (global_kz != kz))[cslices]

# 2D Boussinesq hydrodynamics
problem = de.IVP(domain, variables=['p','b','u','w'])
problem.parameters['N'] = N
problem.parameters['ν'] = ν
problem.parameters['κ'] = κ
problem.parameters['FP'] = FP
problem.parameters['FN'] = FN
problem.substitutions['C'] = "Convolve"
problem.substitutions['KE_P'] = "0.5*(C(FP,u)**2 + C(FP,w)**2 + C(FP,b)**2/N**2)"
problem.substitutions['KE_N'] = "0.5*(C(FN,u)**2 + C(FN,w)**2 + C(FN,b)**2/N**2)"
problem.add_equation("p = 0", condition="(nx == 0) and (nz == 0)")
problem.add_equation("dx(u) + dz(w) = 0", condition="(nx != 0) or (nz != 0)")
problem.add_equation("dt(b) - κ*(dx(dx(b)) + dz(dz(b))) + N**2*w    = -(u*dx(b) + w*dz(b))")
problem.add_equation("dt(u) - ν*(dx(dx(u)) + dz(dz(u))) + dx(p)     = -(u*dx(u) + w*dz(u))")
problem.add_equation("dt(w) - ν*(dx(dx(w)) + dz(dz(w))) + dz(p) - b = -(u*dx(w) + w*dz(w))")

# Build solver
solver = problem.build_solver(de.timesteppers.RK222)
logger.info('Solver built')

# Initial conditions
solver.load_state('restart.h5', -1)

# Integration parameters
solver.stop_sim_time = 4000
solver.stop_wall_time = 30 * 60.
solver.stop_iteration = np.inf
dt = 2**-2

# Analysis
an1 = solver.evaluator.add_file_handler('data_grid', sim_dt=4*np.pi, max_writes=50)
an1.add_system(solver.state, layout='g')

an2 = solver.evaluator.add_file_handler('data_coeff', sim_dt=4*np.pi, max_writes=50)
an2.add_system(solver.state, layout='c')

an3 = solver.evaluator.add_file_handler('data_scalar', iter=10, max_writes=1000)
an3.add_task("integ(KE_P)")
an3.add_task("integ(KE_N)")

# CFL
# CFL = flow_tools.CFL(solver, initial_dt=2**-3, cadence=10, safety=0.5,
#                      max_change=1.5, min_change=0.5, max_dt=2**-2, threshold=0.05)
# CFL.add_velocities(('u', 'w'))

# Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.ok:
        dt = solver.step(dt, trim=False)
        if (solver.iteration-1) % 10 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))
