"""
Ideal fully-compressible hydrodynamics.

"""

import numpy as np
from mpi4py import MPI
import time
import pathlib

from dedalus import public as de
from dedalus.extras import flow_tools
import atmospheres as atmos
import parameters as param

import logging
logger = logging.getLogger(__name__)


# BVP domain
z_basis = de.Chebyshev('z', param.z_res, interval=(0, param.Lz), dealias=3/2)
domain = de.Domain([z_basis], grid_dtype=np.float64, comm=MPI.COMM_SELF)

# Solve BVP for background
X0 = np.array([param.p_bottom, -param.rho_bottom*param.g])
P = (param.N2_func, param.g, param.gamma)
p_bvp, pz_bvp = atmos.solve_dedalus(X0, P, domain, ncc_cutoff=param.ncc_cutoff)

np.seterr(all='raise')

# IVP domain
x_basis = de.Fourier('x', param.x_res, interval=(0, param.Lx), dealias=3/2)
z_basis = de.Chebyshev('z', param.z_res, interval=(0, param.Lz), dealias=3/2)
domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)

# Background state
a0 = domain.new_field()
p0 = domain.new_field()
slices = domain.dist.grid_layout.slices(scales=1)
a0.meta['x']['constant'] = True
p0.meta['x']['constant'] = True
a0['g'][:] = 1 / (- pz_bvp['g'][slices[1]] / param.g)
p0['g'][:] = p_bvp['g'][slices[1]]

# Adiabatic viscous fully-compressible hydrodynamics
problem = de.IVP(domain, variables=['a1','p1','u','w','uz','wz'], ncc_cutoff=param.ncc_cutoff)
problem.meta['w']['z']['dirichlet'] = True
problem.parameters['a0'] = a0
problem.parameters['p0'] = p0
problem.parameters['a0z'] = a0.differentiate('z')
problem.parameters['p0z'] = p0.differentiate('z')
problem.parameters['a0x'] = 0#a0.differentiate('x')
problem.parameters['p0x'] = 0#p0.differentiate('x')
problem.parameters['μ'] = param.mu
problem.parameters['γ'] = param.gamma
problem.parameters['k'] = param.k_tide
problem.parameters['ω'] = param.omega_tide
problem.parameters['A'] = param.A_tide
problem.parameters['Lz'] = param.Lz
problem.substitutions['ux'] = "dx(u)"
problem.substitutions['wx'] = "dx(w)"
problem.substitutions['div_u'] = "ux + wz"
problem.substitutions['txx'] = "μ*(2*ux - 2/3*div_u)"
problem.substitutions['txz'] = "μ*(wx + uz)"
problem.substitutions['tzz'] = "μ*(2*wz - 2/3*div_u)"
problem.substitutions['φ'] = "A*cos(k*x - ω*t)*exp(k*(z - Lz))"
problem.substitutions['cs20'] = "γ*p0*a0"
problem.add_equation("dt(u) + a0*dx(p1) + a1*p0x - a0*(dx(txx) + dz(txz)) = - (u*ux + w*uz) - a1*dx(p1) + a1*(dx(txx) + dz(txz)) - dx(φ)")
problem.add_equation("dt(w) + a0*dz(p1) + a1*p0z - a0*(dx(txz) + dz(tzz)) = - (u*wx + w*wz) - a1*dz(p1) + a1*(dx(txz) + dz(tzz)) - dz(φ)")
problem.add_equation("dt(a1) + u*a0x + w*a0z -   a0*div_u = - (u*dx(a1) + w*dz(a1)) +   a1*div_u")
problem.add_equation("dt(p1) + u*p0x + w*p0z + γ*p0*div_u = - (u*dx(p1) + w*dz(p1)) - γ*p1*div_u")
problem.add_equation("uz - dz(u) = 0")
problem.add_equation("wz - dz(w) = 0")
problem.add_bc("left(uz) = 0")
problem.add_bc("right(uz) = 0")
problem.add_bc("left(w) = 0")
problem.add_bc("right(w) = 0")

# Solver
solver = problem.build_solver(param.ts)
solver.stop_sim_time = param.stop_sim_time
solver.stop_wall_time = param.stop_wall_time
solver.stop_iteration = param.stop_iteration

# Initial conditions
if pathlib.Path('restart.h5').exists():
    write, initial_dt = solver.load_state('restart.h5', -1)
    param.CFL['initial_dt'] = initial_dt
else:
    pass

# Output
# Checkpoints
an0 = solver.evaluator.add_file_handler('snapshots_grid', sim_dt=param.snapshot_sim_dt, max_writes=10)
an0.add_system(solver.state, layout='g')
an1 = solver.evaluator.add_file_handler('snapshots_coeff', sim_dt=param.snapshot_sim_dt, max_writes=10)
an1.add_system(solver.state, layout='c')

# CFL calculator
CFL = flow_tools.CFL(solver, **param.CFL)
CFL.add_velocities(('u', 'w'))

# Main loop
dt_floor = 2**(np.log2(param.CFL['min_dt'])//1)
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.ok:
        dt = CFL.compute_dt()
        dt = dt_floor * (dt // dt_floor)
        dt = solver.step(dt, trim=True)
        if (solver.iteration-1) % param.CFL['cadence'] == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
except:
    logger.error('Exception raised, triggering end of main loop.')
    logger.error('Final timestep: %f' %dt)
    raise
finally:
    end_time = time.time()
    run_time = end_time - start_time
    logger.info('Initial iteration: %i' %solver.initial_iteration)
    logger.info('Initial sim time : %f' %solver.initial_sim_time)
    logger.info('Final iteration  : %i' %solver.iteration)
    logger.info('Final sim time   : %f' %solver.sim_time)
    logger.info('Net iterations   : %i' %(solver.iteration-solver.initial_iteration))
    logger.info('Net sim time     : %f' %(solver.sim_time-solver.initial_sim_time))
    logger.info('Run time: %.2f sec' %run_time)
    logger.info('Run time: %f cpu-hr' %(run_time/60/60*domain.dist.comm_cart.size))
