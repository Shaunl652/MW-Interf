# =============================================================================
# Generating the interferance pattern for a given scenario
# =============================================================================

import numpy as np
import json
from Talbot.Likelihood import Like
import sys
from Experiment import Experiment
import scipy.constants as const
AMU = const.atomic_mass


# Parameter space
# =============================================================================
with open("Space.json","r") as read_file:
    Spc = json.load(read_file)

lam_axis = np.logspace(Spc['Lambda']['Min'],Spc['Lambda']['Max'],Spc['Lambda']['Step'])
r_C_axis = np.logspace(Spc['r_c_ps']['Min'],Spc['r_c_ps']['Max'],Spc['r_c_ps']['Step'])


# Experimental Perameters and main code
# =============================================================================
prm_file = "Scenarios/"+sys.argv[1]+".json"
if len(sys.argv)>2:
    if sys.argv[2]== 'PL' or sys.argv[2]=='pl' or sys.argv[2]=='Point-Like' or sys.argv[2]=='point-like':
        PL = True
    else:
        PL = False
else:
    PL = False

with open(prm_file,"r") as read_file:
    prms = json.load(read_file)

expt = Experiment(sys.argv[1])


# Produce the output array
yaxis = Like(expt,Talbot="Mie",Decoherence=False,CSL=None)

# Plotting
# ============================================================================

import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
if PL:
    ax.set(title=f'Point-Like Interfeance pattern, kR = {expt.k*expt.Radius:.2f}',)
else:
    ax.set(title=f'Mie Theory Interfeance pattern, mass = {expt.mass/AMU:.2e}',)
ax.set(xlabel='$x [\\mu m]$',ylabel='Intensity')
ax.plot(expt.xaxis/1e-6,yaxis)

fig.show()



