"""
pg coupling parameters
"""

import numpy as np
import dedalus.public as de

# Atmosphere
gamma = 5/3
g = 1
p_bottom = 1
rho_bottom = 1
mu = 1e-2

# Domain
Lz = 2
Lx = 4
z_res = 128
x_res = 32
ncc_cutoff = 1e-6

# Cavity
cav_center = Lz / 2
cav_width = Lz / 2 / 3
N2_amp = 0.2**2

def N2_func(z):
    # Gaussian cavity
    zc = (z - cav_center) / cav_width
    return N2_amp * np.exp(-zc**2/2)

# Tide
A_tide = 1e-1
k_tide = 2 * np.pi / (Lx/2)
omega_tide = 0.5

# Stopping
stop_sim_time = 200.0
stop_wall_time = 12*60*60
stop_iteration = np.inf

# Analysis
snapshot_sim_dt = 1.0

# Timestepping
ts = de.timesteppers.RK222
CFL = {'initial_dt': 10**-1,
       'min_dt': 10**-4,
       'max_dt': 10**-1,
       'safety': 0.5,
       'cadence': 10,
       'min_change': 0.5,
       'max_change': 1.5,
       'threshold': 0.05}
