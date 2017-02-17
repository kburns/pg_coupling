"""
Ideal fully-compressible hydrodynamics.

"""

import numpy as np
from mpi4py import MPI
import time
from dedalus import public as de
from dedalus.extras import flow_tools
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import atmospheres as atmos
plt.ioff()

import logging
logger = logging.getLogger(__name__)


# Parameters
γ = 5/3
g = 1
p_bottom = 1
ρ_bottom = 1
Lz = 2
zcb = 0.75
zct = 1.25
kx = 1
N2_amp = 0.2**2
τ = 2*np.pi/0.5

def N2_func(z):
    # Quadratic buoyancy profile in cavity
    zc = (z - zcb) / (zct - zcb)
    if 0 < zc < 1:
        return N2_amp * zc * (1 - zc) / (1 / 2)**2
    else:
        return 0

def N2_func(z):
    # Gaussian cavity
    zc0 = (zcb + zct) / 2
    zcw = (zct - zcb) / 2 / 3
    zc = (z - zc0) / zcw
    return N2_amp * np.exp(-zc**2/2)

# Discretization parameters
Nz = 128

# Create bases and domain
z_basis = de.Chebyshev('z', Nz, interval=(0, Lz), dealias=3/2)
domain = de.Domain([z_basis], grid_dtype=np.complex128)
z = domain.grid(0)
zd = domain.grid(0, scales=domain.dealias)

# Solve BVP for background
X0 = np.array([p_bottom, -ρ_bottom*g])
P = (N2_func, g, γ)
p_bvp, pz_bvp = atmos.solve_dedalus(X0, P, domain, ncc_cutoff=1e-6)

# Background state
α0 = domain.new_field(scales=domain.dealias)
p0 = domain.new_field(scales=domain.dealias)
α0['g'] = 1 / (- pz_bvp['g'] / g)
p0['g'] = p_bvp['g']

# Sponge
s = domain.new_field(scales=domain.dealias)
#s['g'] = (1 + np.tanh((zd - (Lz+Ls/2))/(Ls/3))) / 2 / τ

# Ideal fully-compressible hydrodynamics
problem = de.EVP(domain, variables=['α','p','u','w','wz'], eigenvalue='ω',
    ncc_cutoff=1e-6)
problem.meta['w']['z']['dirichlet'] = True
problem.parameters['kx'] = kx
problem.parameters['α0'] = α0
problem.parameters['p0'] = p0
problem.parameters['α0z'] = α0.differentiate('z')
problem.parameters['p0z'] = p0.differentiate('z')
problem.parameters['γ'] = γ
problem.parameters['s'] = 0
problem.substitutions['dt(A)'] = "-1j*ω*A"
problem.substitutions['dx(A)'] = "1j*kx*A"
problem.add_equation("dt(u) + α0*dx(p)         + s*u = 0")
problem.add_equation("dt(w) + α0*dz(p) + α*p0z + s*w = 0")
problem.add_equation("dt(α) + w*α0z -   α0*(dx(u) + wz) + s*α = 0")
problem.add_equation("dt(p) + w*p0z + γ*p0*(dx(u) + wz) + s*p = 0")
problem.add_equation("wz - dz(w) = 0")
problem.add_bc("left(w) = 0")
problem.add_bc("right(w) = 0")

# Build solver
solver = problem.build_solver()
solver.solve(solver.pencils[0])

# Filter eigenvalues
evals = solver.eigenvalues.copy()
evals = evals[np.isfinite(evals)]
evals = evals[np.abs(evals) < 0.5]
evals = evals[np.abs(evals) > 0.01]
# evals = evals[np.abs(evals) < 10]
# evals = evals[np.abs(evals) > 0.5]

# index1 = np.argmax(evals.real)
# index2 = np.where(solver.eigenvalues == evals[index1])[0][0]
# solver.set_state(index2)

fig = plt.figure()
axes = fig.add_subplot(1,1,1)
axes.plot(zd, p0['g'], label='p0')
axes.plot(zd, α0['g'], label='α0')
axes.plot(zd, s['g']*τ, label='sτ')
axes.legend()
fig.savefig('background.pdf')

fig = plt.figure(figsize=(12,8))
plt.clf()
plt.scatter(evals.real, evals.imag)
plt.savefig('evals.pdf')

plt.clf()
fig = plt.gcf()
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)
scales = 4
α0.set_scales(scales)
for field in solver.state.fields:
    field.set_scales(scales)
z = domain.grid(0, scales=scales)
for ev in evals:
    print(ev)
    index = np.where(solver.eigenvalues == ev)[0][0]
    solver.set_state(index)
    u = solver.state['u']['g']
    w = solver.state['w']['g']
    p = solver.state['p']['g']
    α = solver.state['α']['g']
    ρ = -α / α0['g']**2
    ax1.plot(z, np.abs(u))
    ax1.set_title('|u|')
    ax2.plot(z, np.abs(w))
    ax2.set_title('|w|')
    ax3.plot(z, np.abs(p))
    ax3.set_title('|p|')
    ax4.plot(z, np.abs(ρ))
    ax4.set_title('|ρ|')
plt.savefig('evecs.pdf')

plt.clf()
plt.semilogy(np.abs(α0['c']),'.-')
plt.savefig('alpha_coeffs.pdf')

# plt.clf()
# plt.plot(z, np.sqrt(γ/4/H), label='ω_a')
# plt.plot(z, np.sqrt((1/Γ-1/γ)/H), label='N')
# plt.legend(loc='upper right')
# plt.savefig('freqs.pdf')
