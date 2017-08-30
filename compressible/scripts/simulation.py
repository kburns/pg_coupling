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
import tides

import logging
logger = logging.getLogger(__name__)


# Solve background
domain, p_bvp = atmos.solve_hydrostatic_pressure(param, np.float64)
p_full, p_trunc, a_full, a_trunc, heq, N2 = atmos.truncate_background(param, p_bvp)
np.seterr(all='raise')

# IVP domain
x_basis = de.Fourier('x', param.Nx, interval=(0, param.Lx), dealias=3/2)
z_basis = de.Chebyshev('z', param.Nz, interval=(0, param.Lz), dealias=3/2)
domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)

# Background state
a0 = domain.new_field()
p0 = domain.new_field()
a0.meta['x']['constant'] = True
p0.meta['x']['constant'] = True
a0.set_scales(1)
p0.set_scales(1)
a_trunc.set_scales(1)
p_trunc.set_scales(1)
slices = domain.dist.grid_layout.slices(scales=1)
a0['g'][:] = a_trunc['g'][slices[1]]
p0['g'][:] = p_trunc['g'][slices[1]]

# Adiabatic viscous fully-compressible hydrodynamics
problem = de.IVP(domain, variables=['a1','p1','u','w','uz','wz'],
    ncc_cutoff=param.ivp_cutoff, entry_cutoff=param.matrix_cutoff)
problem.meta[:]['z']['dirichlet'] = True
problem.parameters['a0'] = a0
problem.parameters['p0'] = p0
problem.parameters['a0z'] = a0.differentiate('z')
problem.parameters['p0z'] = p0.differentiate('z')
problem.substitutions['a0x'] = '0' #a0.differentiate('x')
problem.substitutions['p0x'] = '0' #p0.differentiate('x')
problem.parameters['U'] = param.U
problem.parameters['μ'] = param.μ
problem.parameters['γ'] = param.γ
problem.parameters['k'] = param.k_tide
problem.parameters['ω'] = param.ω_tide
problem.parameters['σ'] = param.σ_tide
problem.parameters['A'] = param.A_tide
problem.parameters['Lz'] = param.Lz
problem.substitutions['ux'] = "dx(u)"
problem.substitutions['wx'] = "dx(w)"
problem.substitutions['div_u'] = "ux + wz"
problem.substitutions['txx'] = "μ*(2*ux - 2/3*div_u)"
problem.substitutions['txz'] = "μ*(wx + uz)"
problem.substitutions['tzz'] = "μ*(2*wz - 2/3*div_u)"
problem.substitutions['φ'] = "A*exp(σ*t)*cos(k*x)*exp(k*(z - Lz))"
problem.substitutions['cs20'] = "γ*p0*a0"
problem.add_equation("dt(u) + U*ux + a0*dx(p1) + a1*p0x - a0*(dx(txx) + dz(txz)) = - (u*ux + w*uz) - a1*dx(p1) + a1*(dx(txx) + dz(txz)) - dx(φ)")
problem.add_equation("dt(w) + U*wx + a0*dz(p1) + a1*p0z - a0*(dx(txz) + dz(tzz)) = - (u*wx + w*wz) - a1*dz(p1) + a1*(dx(txz) + dz(tzz)) - dz(φ)")
problem.add_equation("dt(a1) + U*dx(a1) + u*a0x + w*a0z -   a0*div_u = - (U*a0x + u*dx(a1) + w*dz(a1)) +   a1*div_u")
problem.add_equation("dt(p1) + U*dx(p1) + u*p0x + w*p0z + γ*p0*div_u = - (U*p0x + u*dx(p1) + w*dz(p1)) - γ*p1*div_u")
problem.add_equation("uz - dz(u) = 0")
problem.add_equation("wz - dz(w) = 0")
problem.add_bc("left(txz) = 0")
problem.add_bc("right(txz) = 0")
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
    _, linear_problem = tides.linear_tide_2d(param)
    linear_solver = linear_problem.build_solver()
    linear_solver.solve()
    for var in problem.variables:
        solver.state[var]['c'] = linear_solver.state[var]['c']

# Output
an0 = solver.evaluator.add_file_handler('data_checkpoints', wall_dt=param.checkpoint_wall_dt, max_writes=1)
an0.add_system(solver.state, layout='c')
an1 = solver.evaluator.add_file_handler('data_snapshots', sim_dt=param.snapshot_sim_dt, max_writes=10)
an1.add_task('u', layout='g')
an1.add_task('w', layout='g')
an1.add_task('p1', layout='g')
an1.add_task('a1', layout='g')
an1.add_task('p0+p1', name='p', layout='g')
an1.add_task('a0+a1', name='a', layout='g')
an1.add_task('(a0+a1)**(-1)', name='ρ', layout='g')
an2 = solver.evaluator.add_file_handler('data_scalars', sim_dt=param.scalar_sim_dt, max_writes=100)
an2.add_task('integ((u*u+w*w)/(a0+a1)/2)', name='KE', layout='g')
an2.add_task('integ(u*dx(txx) + u*dz(txz) + w*dx(txz) + w*dz(tzz))', name='D', layout='g')

# Monitoring
flow = flow_tools.GlobalFlowProperty(solver, cadence=param.CFL['cadence'])
flow.add_property('integ((u*u+w*w)/(a0+a1)/2)', name='KE')

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
        dt = solver.step(dt, trim=param.trim)
        if (solver.iteration-1) % param.CFL['cadence'] == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
            logger.info('Ave KE = %e' %flow.max('KE'))
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
