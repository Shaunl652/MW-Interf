# =============================================================================
# Arguments should be  SCENARIO and PRIOR TYPE
# =============================================================================

import numpy as np
#import sys
import matplotlib.pyplot as plt
from pathlib import Path



# First we set up the path where to save the graphs
Grp_Path = "Plots/Info/Data"
Path(Grp_Path).mkdir(parents=True, exist_ok=True) 


# Next we read in all the data
# Read in Expt stuff first
NPY_Path = "NPYs/MAQRO8_Opt/Experimental/Data"

xaxis = [*range(0,16000,200)]

ExptMCdata = np.load(f'{NPY_Path}/InfoMCMC.npy') # <H>_theta
ExptMCstd = np.load(f'{NPY_Path}/Err_MCMC.npy') # Errors in <H>_theta
ExptMCup_bound = ExptMCdata + np.sqrt(ExptMCstd)
ExptMClo_bound = ExptMCdata - np.sqrt(ExptMCstd)


# Next we laod and plot the data from P(X|theta=0)

ExptTHdata = np.load(f'{NPY_Path}/InfoTheta.npy')
ExptTHstd = np.load(f'{NPY_Path}/Err_Theta.npy')
ExptTHup_bound = ExptTHdata + np.sqrt(ExptTHstd)
ExptTHlo_bound = ExptTHdata - np.sqrt(ExptTHstd)

# Then the MDIP
NPY_Path = "NPYs/MAQRO8_Opt/MDIP/Data"

MDIPMCdata = np.load(f'{NPY_Path}/InfoMCMC.npy') # <H>_theta
MDIPMCstd = np.load(f'{NPY_Path}/Err_MCMC.npy') # Errors in <H>_theta
MDIPMCup_bound = MDIPMCdata + np.sqrt(MDIPMCstd)
MDIPMClo_bound = MDIPMCdata - np.sqrt(MDIPMCstd)


# Next we laod and plot the data from P(X|theta=0)

MDIPTHdata = np.load(f'{NPY_Path}/InfoTheta.npy')
MDIPTHstd = np.load(f'{NPY_Path}/Err_Theta.npy')
MDIPTHup_bound = MDIPTHdata + np.sqrt(MDIPTHstd)
MDIPTHlo_bound = MDIPTHdata - np.sqrt(MDIPTHstd)


# Now we can start plotting

fig,axes = plt.subplots()

# First plot the experimental then MDIP
# Do MCMC First
# axes[0].plot(xaxis,ExptMCdata,label='Experimental Prior')
# axes[0].fill_between(xaxis, ExptMCup_bound,ExptMClo_bound, alpha=0.2)

# axes[0].plot(xaxis,MDIPMCdata,label='MDIP')
# axes[0].fill_between(xaxis, MDIPMCup_bound,MDIPMClo_bound, alpha=0.2)

# axes[0].set(title='a)',xlabel='Number of Data Points')

# Now we do the one where theta = 0
axes.plot(xaxis,ExptTHdata,label='Experimental Prior')
axes.fill_between(xaxis, ExptTHup_bound,ExptTHlo_bound, alpha=0.2)

axes.plot(xaxis,MDIPTHdata,label='MDIP')
axes.fill_between(xaxis, MDIPTHup_bound,MDIPTHlo_bound, alpha=0.2)

axes.set(xlabel='Number of Data Points',ylabel='$ \\langle \\mathcal{H} \\rangle $')
axes.grid(which='both')
axes.legend()
# h,l = axes[0].get_legend_handles_labels()

# fig.legend(h,l)

fig.tight_layout()

fig.savefig(f'{Grp_Path}/Info_Mass.png')

fig.savefig(f'{Grp_Path}/Info_Mass.pdf')




    