#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Plots the Info or lambda as a funtion of mass
# Arguments should be  SCENARIO and PRIOR TYPE
# =============================================================================

import numpy as np
#import sys
import matplotlib.pyplot as plt
from pathlib import Path
plt.rcParams.update({'font.size': 15})


# First we set up the path where to save the graphs
Grp_Path = "Plots/Info/Drifts"
Path(Grp_Path).mkdir(parents=True, exist_ok=True) 

Drift_vals = ['01','1','10','30','100']
labels = ['0.1 nm','1 nm','10 nm','30 nm','100 nm']
fig,axes = plt.subplots()

for label,DV in zip(labels,Drift_vals):

    # Next we read in all the data
    # Read in Expt stuff first
    NPY_Path = f"NPYs/Drift{DV}/MDIP/Mass"
    
    
    # Next we laod and plot the data from P(X|theta=0)
    xaxis = np.load(f'{NPY_Path}/Mass.npy')
    data  = np.load(f'{NPY_Path}/Info_vals_THETA.npy')
    std   = np.load(f'{NPY_Path}/Info_Vars_THETA.npy')
    up_bound = data + np.sqrt(std)
    lo_bound = data - np.sqrt(std)
    
    
    # Now we can start plotting
    
    
    
    # First plot the experimental then MDIP
    # Do MCMC First
    axes.plot(xaxis,data,label=label)
    axes.fill_between(xaxis, up_bound,lo_bound, alpha=0.2)
    
    axes.set(xlabel='Mass [u]',xscale='log',ylabel='$ \\langle \\mathcal{H} \\rangle $')
    axes.grid(which='both')
    
h,l = axes.get_legend_handles_labels()

fig.legend(h,l)

fig.tight_layout()

fig.savefig(f'{Grp_Path}/Fig7.png')

fig.savefig(f'{Grp_Path}/Fig7.pdf')
