"""
Ideal fully-compressible hydrodynamics.

"""

import numpy as np
from mpi4py import MPI
import time
import pathlib

from dedalus import public as de
from dedalus.extras import flow_tools
import background as atmos
import parameters as param
import tides

import logging
logger = logging.getLogger(__name__)


# Solve 2d linear problem
atmos, linear_problem = tides.linear_tide_2d(param)
linear_solver = linear_problem.build_solver()
linear_solver.solve()

# Adiabatic viscous fully-compressible hydrodynamics
problem = de.IVP(atmos.domain, variables=['a2','p2','u2','w2','u2z','w2z'],
    ncc_cutoff=param.ivp_cutoff, entry_cutoff=param.matrix_cutoff)
problem.meta[:]['z']['dirichlet'] = True
problem.parameters['a1'] = linear_solver.state['a1']
problem.parameters['p1'] = linear_solver.state['p1']
problem.parameters['u1'] = linear_solver.state['u']
problem.parameters['w1'] = linear_solver.state['w']
problem.parameters['a0'] = atmos.a0
problem.parameters['p0'] = atmos.p0
problem.parameters['a0z'] = atmos.a0z
problem.parameters['p0z'] = atmos.p0z
problem.parameters['U'] = param.U
problem.parameters['μ'] = param.μ
problem.parameters['γ'] = param.γ
problem.parameters['k'] = param.k_tide
problem.parameters['ω'] = param.ω_tide
problem.parameters['σ'] = param.σ_tide
problem.parameters['A'] = param.A_tide
problem.parameters['Lz'] = param.Lz
problem.substitutions['a0x'] = '0'
problem.substitutions['p0x'] = '0'
problem.substitutions['u1x'] = "dx(u1)"
problem.substitutions['w1x'] = "dx(w1)"
problem.substitutions['u2x'] = "dx(u2)"
problem.substitutions['w2x'] = "dx(w2)"
problem.substitutions['u1z'] = "dz(u1)"
problem.substitutions['w1z'] = "dz(w1)"
problem.substitutions['div_u1'] = "u1x + w1z"
problem.substitutions['div_u2'] = "u2x + w2z"
problem.substitutions['t1xx'] = "μ*(2*u1x - 2/3*div_u1)"
problem.substitutions['t1xz'] = "μ*(w1x + u1z)"
problem.substitutions['t1zz'] = "μ*(2*w1z - 2/3*div_u1)"
problem.substitutions['t2xx'] = "μ*(2*u2x - 2/3*div_u2)"
problem.substitutions['t2xz'] = "μ*(w2x + u2z)"
problem.substitutions['t2zz'] = "μ*(2*w2z - 2/3*div_u2)"
problem.substitutions['φ'] = "A*exp(σ*t)*cos(k*x)*exp(k*(z - Lz))"
problem.substitutions['cs20'] = "γ*p0*a0"
problem.add_equation("dt(u2) + U*u2x + a0*dx(p2) + a2*p0x - a0*(dx(t2xx) + dz(t2xz)) = - ((u1+u2)*(u1x+u2x) + (w1+w2)*(u1z+u2z)) - (a1+a2)*dx((p1+p2)) + (a1+a2)*(dx(t1xx+t2xx) + dz(t1xz+t2xz))")
problem.add_equation("dt(w2) + U*w2x + a0*dz(p2) + a2*p0z - a0*(dx(t2xz) + dz(t2zz)) = - ((u1+u2)*(w1x+w2x) + (w1+w2)*(w1z+w2z)) - (a1+a2)*dz((p1+p2)) + (a1+a2)*(dx(t1xz+t2xz) + dz(t1zz+t2zz))")
problem.add_equation("dt(a2) + U*dx(a2) + u2*a0x + w2*a0z -   a0*div_u2 = - (U*a0x + (u1+u2)*dx((a1+a2)) + (w1+w2)*dz((a1+a2))) +   (a1+a2)*(div_u1+div_u2)")
problem.add_equation("dt(p2) + U*dx(p2) + u2*p0x + w2*p0z + γ*p0*div_u2 = - (U*p0x + (u1+u2)*dx((p1+p2)) + (w1+w2)*dz((p1+p2))) - γ*(p1+p2)*(div_u1+div_u2)")
problem.add_equation("u2z - dz(u2) = 0")
problem.add_equation("w2z - dz(w2) = 0")
problem.add_bc("left(t2xz) = 0")
problem.add_bc("right(t2xz) = 0")
problem.add_bc("left(w2) = 0")
problem.add_bc("right(w2) = 0")

# Solver
solver = problem.build_solver(param.ts)
solver.stop_sim_time = param.stop_sim_time
solver.stop_wall_time = param.stop_wall_time
solver.stop_iteration = param.stop_iteration

# Initial conditions
if pathlib.Path('restart.h5').exists():
    write, initial_dt = solver.load_state('restart.h5', -1)
    param.CFL['initial_dt'] = initial_dt

# Output
an0 = solver.evaluator.add_file_handler('data_checkpoints', wall_dt=param.checkpoint_wall_dt, max_writes=1)
an0.add_system(solver.state, layout='c')
an3 = solver.evaluator.add_file_handler('data_snapshots_coeff', sim_dt=param.snapshot_sim_dt, max_writes=10)
an3.add_system(solver.state, layout='c')
an1 = solver.evaluator.add_file_handler('data_snapshots', sim_dt=param.snapshot_sim_dt, max_writes=10)
an1.add_system(solver.state, layout='g')
an1.add_task('p0+p1+p2', name='p', layout='g')
an1.add_task('a0+a1+p2', name='a', layout='g')
an1.add_task('(a0+a1+a2)**(-1)', name='ρ', layout='g')
an1.add_task('a1+a2', name='diff_a1', layout='g')
an1.add_task('p1+p2', name='diff_p1', layout='g')
an1.add_task('u1+u2', name='diff_u', layout='g')
an1.add_task('w1+w2', name='diff_w', layout='g')
an2 = solver.evaluator.add_file_handler('data_scalars', sim_dt=param.scalar_sim_dt, max_writes=100)
an2.add_task('integ(((u1+u2)**2 + (w1+w2)**2) /(a0+a1+a2)/2)', name='KE', layout='g')
an2.add_task('integ((u2*u2+w2*w2)/(a0+a1+a2)/2)', name='KE_pert', layout='g')
an2.add_task('-integ(u*dx(txx) + u*dz(txz) + w*dx(txz) + w*dz(tzz))', name='D', layout='g')

# Monitoring
flow = flow_tools.GlobalFlowProperty(solver, cadence=param.CFL['cadence'])
flow.add_property('integ((u2*u2+w2*w2)/(a0+a1+a2)/2)', name='KE_pert')

# CFL calculator
CFL = flow_tools.CFL(solver, **param.CFL)
CFL.add_velocities(('u1+u2', 'w1+w2'))

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
            logger.info('Ave KE = %e' %flow.max('KE+pert'))
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
