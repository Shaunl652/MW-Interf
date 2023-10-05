import numpy as np
from Experiment import Experiment
from Talbot.Grating.Coefficients import Coef

expt = Experiment("Belenchia_fig5")

phi0s = np.linspace(0,12,51)
phi0s[0] += 1e-6

VisRay = list(); VisMie = list()
VisRayNoAbs = list(); VisMieNoAbs = list()

for idx,phi0 in enumerate(phi0s):
    print(f"{idx}/{len(phi0s)} : {phi0:.2f}")
    expt.phi0 = phi0
    
    expt.grating_photon_absorption = True
    B = Coef(1,expt,Talbot="Rayleigh"); VisRay.append(2*abs(B[1]))
    B = Coef(1,expt,Talbot="Mie");      VisMie.append(2*abs(B[1]))

    expt.grating_photon_absorption = False
    B = Coef(1,expt,Talbot="Rayleigh"); VisRayNoAbs.append(2*abs(B[1]))
    B = Coef(1,expt,Talbot="Mie");      VisMieNoAbs.append(2*abs(B[1]))

import matplotlib.pyplot as plt
plt.figure()
plt.plot(phi0s,VisRay,        'b',label="Rayleigh")
plt.plot(phi0s,VisRayNoAbs, '--b',label="Rayleigh n abs")
plt.plot(phi0s,VisMie,        'r',label="Mie")
plt.plot(phi0s,VisMieNoAbs, '--r',label="Mie no abs")
plt.ylim(top=0.6)

plt.legend()
plt.show()
