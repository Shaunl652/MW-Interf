# =============================================================================
# Plots the Info or lambda as a funtion of mass
# Arguments should be  SCENARIO and PRIOR TYPE
# =============================================================================

import numpy as np
#import sys
import matplotlib.pyplot as plt
from pathlib import Path



# First we set up the path where to save the graphs
Grp_Path = "Plots/Info/Mass"
Path(Grp_Path).mkdir(parents=True, exist_ok=True) 


# Next we read in all the data
# Read in Expt stuff first
NPY_Path = "NPYs/MAQRO8_Opt/Experimental/Mass"

ExptMCxaxis = np.load(f'{NPY_Path}/Mass.npy') # Masses where we measured <H>_theta
ExptMCdata = np.load(f'{NPY_Path}/InfoMCMC.npy') # <H>_theta
ExptMCstd = np.load(f'{NPY_Path}/ErrsMCMC.npy') # Errors in <H>_theta
ExptMCup_bound = ExptMCdata + np.sqrt(ExptMCstd)
ExptMClo_bound = ExptMCdata - np.sqrt(ExptMCstd)


# Next we laod and plot the data from P(X|theta=0)
ExptTHxaxis = np.load(f'{NPY_Path}/MassTheta.npy')
ExptTHdata = np.load(f'{NPY_Path}/InfoTheta.npy')
ExptTHstd = np.load(f'{NPY_Path}/ErrsTheta.npy')
ExptTHup_bound = ExptTHdata + np.sqrt(ExptTHstd)
ExptTHlo_bound = ExptTHdata - np.sqrt(ExptTHstd)

# Then the MDIP
NPY_Path = "NPYs/MAQRO8_Opt/MDIP/Mass"
MDIPMCxaxis = np.load(f'{NPY_Path}/Mass.npy') # Masses where we measured <H>_theta
MDIPMCdata = np.load(f'{NPY_Path}/InfoMCMC.npy') # <H>_theta
MDIPMCstd = np.load(f'{NPY_Path}/ErrsMCMC.npy') # Errors in <H>_theta
MDIPMCup_bound = MDIPMCdata + np.sqrt(MDIPMCstd)
MDIPMClo_bound = MDIPMCdata - np.sqrt(MDIPMCstd)


# Next we laod and plot the data from P(X|theta=0)
MDIPTHxaxis = np.load(f'{NPY_Path}/MassTheta.npy')
MDIPTHdata = np.load(f'{NPY_Path}/InfoTheta.npy')
MDIPTHstd = np.load(f'{NPY_Path}/ErrsTheta.npy')
MDIPTHup_bound = MDIPTHdata + np.sqrt(MDIPTHstd)
MDIPTHlo_bound = MDIPTHdata - np.sqrt(MDIPTHstd)


# Now we can start plotting

fig,axes = plt.subplots(ncols=2,figsize=(10,5))

# First plot the experimental then MDIP
# Do MCMC First
axes[0].plot(ExptMCxaxis,ExptMCdata,label='Experimental Prior')
axes[0].fill_between(ExptMCxaxis, ExptMCup_bound,ExptMClo_bound, alpha=0.2)

axes[0].plot(MDIPMCxaxis,MDIPMCdata,label='MDIP')
axes[0].fill_between(MDIPMCxaxis, MDIPMCup_bound,MDIPMClo_bound, alpha=0.2)

axes[0].set(title='(a)',xlabel='Mass [u]',xscale='log',ylabel='$ \\langle \\mathcal{H} \\rangle $')
axes[0].grid(which='both')
# Now we do the one where theta = 0
axes[1].plot(ExptTHxaxis,ExptTHdata,label='Experimental Prior')
axes[1].fill_between(ExptTHxaxis, ExptTHup_bound,ExptTHlo_bound, alpha=0.2)

axes[1].plot(MDIPTHxaxis,MDIPTHdata,label='MDIP')
axes[1].fill_between(MDIPTHxaxis, MDIPTHup_bound,MDIPTHlo_bound, alpha=0.2)

axes[1].set(title='(b)',xlabel='Mass [u]',xscale='log',ylabel='$ \\langle \\mathcal{H} \\rangle $')
axes[1].grid(which='both')
h,l = axes[0].get_legend_handles_labels()

fig.legend(h,l)

fig.tight_layout()

fig.savefig(f'{Grp_Path}/Info_Mass.png')

fig.savefig(f'{Grp_Path}/Info_Mass.pdf')




    