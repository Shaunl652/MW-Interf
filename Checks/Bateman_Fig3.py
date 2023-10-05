# =============================================================================
# Produces a graph of the visability of the interferance pattern wrt the phase
# =============================================================================

import numpy as np
from Experiment import Experiment
from Talbot.Grating.Coefficients import Coef

expt = Experiment("Bateman_fig3")

pi = np.pi
phi0s = np.linspace(0,4*pi,51)
phi0s[0] += 1e-6

VisRay = list()
VisRayNoAbs = list()

for idx,phi0 in enumerate(phi0s):
    print(f"{idx}/{len(phi0s)} : {phi0:.2f}")
    expt.phi0 = phi0
    
    expt.grating_photon_absorption = True
    B = Coef(1,expt,Talbot="Rayleigh"); VisRay.append(2*abs(B[1]))

    expt.grating_photon_absorption = False
    B = Coef(1,expt,Talbot="Rayleigh"); VisRayNoAbs.append(2*abs(B[1]))

import matplotlib.pyplot as plt
plt.figure()
plt.plot(phi0s/pi,VisRay,        'b',label="Rayleigh")
plt.plot(phi0s/pi,VisRayNoAbs, '--b',label="Rayleigh without absorption")
plt.xlabel("Phase [pi]")
plt.legend()
plt.show()
