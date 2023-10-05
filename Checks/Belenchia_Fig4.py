import numpy as np
pi = np.pi
exp = np.exp
sqrt = np.sqrt
from Experiment import Experiment

expt = Experiment("Belenchia_fig4")
expt.phi0 = 4.0

from Talbot.Likelihood import Like

wRay = Like(expt,Talbot="Rayleigh",Decoherence=False,CSL=None)
wMie = Like(expt,Talbot="Mie",     Decoherence=False,CSL=None)

import matplotlib.pyplot as plt
plt.plot(wRay,label="Rayleigh")
plt.plot(wMie,label="Mie")
plt.legend()
plt.show()
