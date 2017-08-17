"""
pg coupling parameters
"""

import numpy as np
import dedalus.public as de

# Atmosphere
γ = 5/3
g = 1
p_bottom = 1
ρ_bottom = 1
μ = 1e-6

# Domain
Lz = 2
Lx = 4
Nz = 128
Nx = 128

# Tolerances
nlbvp_cutoff = 1e-9
nlbvp_max_terms = 64
nlbvp_tolerance = 1e-9
pressure_floor = 1e-12
background_floor = 1e-9
ivp_cutoff = 1e-9
matrix_cutoff = 1e-11

# Cavity
cav_center = Lz / 2
cav_width = Lz / 2 / 5
N2_amp = 0.2**2

def N2_func(z):
    # Gaussian cavity
    zc = (z - cav_center) / cav_width
    return N2_amp * np.exp(-zc**2/2)

# Tide
A_tide = 1e-6
k_tide = 2 * np.pi / (Lx/2)
ω_tide = 0.02874588
σ_tide = 0

# Boost to tidal frame
U = - ω_tide / k_tide
ω_tide = 0

# Stopping
stop_sim_time = 15000.0
stop_wall_time = 10*60*60
stop_iteration = np.inf

# Analysis
checkpoint_wall_dt = 30*60
snapshot_sim_dt = 1.0
scalar_sim_dt = 1.0


# Timestepping
ts = de.timesteppers.RK443
CFL = {'initial_dt': 10**-1,
       'min_dt': 10**-4,
       'max_dt': 10**-1,
       'safety': 0.5,
       'cadence': 10,
       'min_change': 0.5,
       'max_change': 1.5,
       'threshold': 0.05}
