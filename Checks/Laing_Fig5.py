#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
# Plots the value of lambda_c at r_c = 1e-7m for different masses
# =============================================================================

import numpy as np
#import sys
import matplotlib.pyplot as plt
from pathlib import Path
plt.rcParams.update({'font.size': 12})


# First we set up the path where to save the graphs
Grp_Path = "Plots/Lambda/"
Path(Grp_Path).mkdir(parents=True, exist_ok=True) 

NPY_Path = f"NPYs/MAQRO8/Experimental/Mass"


# Next we laod the data for the experimental prior
Exptxaxis = np.load(f'{NPY_Path}/Mass.npy')
Exptdata  = np.load(f'{NPY_Path}/Lamb_Vals_THETA.npy')
Exptstd   = np.load(f'{NPY_Path}/Lamb_Vars_THETA.npy')
Exptup_bound = Exptdata + np.sqrt(Exptstd)
Exptlo_bound = Exptdata - np.sqrt(Exptstd)

# Now the MDIP
NPY_Path = f"NPYs/MAQRO8/MDIP/Mass"

MDIPxaxis = np.load(f'{NPY_Path}/Mass.npy')
MDIPdata  = np.load(f'{NPY_Path}/Lamb_Vals_THETA.npy')
MDIPstd   = np.load(f'{NPY_Path}/Lamb_Vars_THETA.npy')
MDIPup_bound = MDIPdata + np.sqrt(MDIPstd)
MDIPlo_bound = MDIPdata - np.sqrt(MDIPstd)

# Now we plot

fig,axes = plt.subplots()
    
# First plot the experimental then MDIP
# Do Expt First
axes.plot(Exptxaxis,Exptdata,label='Experimental Prior')
axes.fill_between(Exptxaxis, Exptup_bound,Exptlo_bound, alpha=0.2)

axes.plot(MDIPxaxis,MDIPdata,label='MDIP')
axes.fill_between(MDIPxaxis,MDIPup_bound,MDIPlo_bound, alpha=0.2)

axes.set(xlabel='Mass [u]',xscale='log',ylabel='$ \\lambda_c $ [Hz]',yscale='log')
axes.grid(which='both')

h,l = axes.get_legend_handles_labels()

fig.legend(h,l)

fig.tight_layout()

fig.savefig(f'{Grp_Path}/Fig5.png')

fig.savefig(f'{Grp_Path}/Fig5.pdf')


