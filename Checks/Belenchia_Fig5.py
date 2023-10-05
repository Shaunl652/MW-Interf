import numpy as np
from Experiment import Experiment
from Talbot.Likelihood import Like

expt = Experiment("Belenchia_fig5")

expt.grating_photon_absorption = True
wRay = Like(expt,Talbot="Rayleigh",Decoherence=False,CSL=None)
wMie = Like(expt,Talbot="Mie",     Decoherence=False,CSL=None)

expt.grating_photon_absorption = False
wRayNoAbs = Like(expt,Talbot="Rayleigh",Decoherence=False,CSL=None)
wMieNoAbs = Like(expt,Talbot="Mie",     Decoherence=False,CSL=None)

import matplotlib.pyplot as plt
plt.plot(wRay,      'b',   label="Rayleigh")
plt.plot(wRayNoAbs, '--b', label="Rayleigh no abs")
plt.plot(wMie,      'r',   label="Mie")
plt.plot(wMieNoAbs, '--r', label="Mie no abs")
plt.legend()
plt.show()
