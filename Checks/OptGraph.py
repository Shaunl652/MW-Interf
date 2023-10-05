#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
# Quick bit of code designed to see how the optimum values of t2 and phi0 change
# as we increase the mass of the particle
# =============================================================================

import numpy as np
from Experiment import Experiment
from Optomise import OptFunc
from scipy.constants import atomic_mass as AMU

expt = Experiment('MAQRO8')

Masses = np.logspace(6,10,101)

PhiPlot = []
t_2Plot = []
for idx,M in enumerate(Masses):
    print(f'\r{idx+1:03}/{len(Masses)} : {M:.3e}',end='')
    expt.mass = M*AMU
    phi,t_2 = OptFunc(expt,idx_plot=idx)
    PhiPlot.append(phi)
    t_2Plot.append(t_2)
    
import matplotlib.pyplot as plt

fig, ax1 = plt.subplots()

colour = 'tab:red'
ax1.set_xlabel('Mass [u]')
ax1.set_ylabel('Opt. $\\Phi_0$',color=colour)
ax1.set_xscale('log')
ax1.plot(Masses,PhiPlot,color=colour)
ax1.tick_params(axis='y',labelcolor=colour)
ax1.grid(which='both')

ax2 = ax1.twinx()

colour = 'tab:blue'
ax2.set_ylabel('Opt. $t_2/t_T$',color=colour)
ax2.set_xscale('log')
ax2.plot(Masses,t_2Plot,color=colour)
ax2.tick_params(axis='y',labelcolor=colour)

fig.tight_layout()
plt.show()