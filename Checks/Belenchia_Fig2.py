# Reproduce Fig2 from Belenchia PRA 100 033813 (2019)
import numpy as np
from Talbot.MieForce import F_z
from scipy.constants import atomic_mass

e = 5.656 + 2.952j
w = 354e-9
k = 2*np.pi/w
z = -w/8

kR = np.logspace(-1,np.log10(4),1001)
R = kR/k
Ms = 2329*np.pi*R**3 * (4/3)/atomic_mass

F = np.array([F_z(z,k,Ri,e) for Ri in R])

Re = np.real
chi = 4*np.pi*R**3*(e-1)/(e+2)
FRay = Re(chi)*k**3

import matplotlib.pyplot as plt
plt.figure()
plt.plot(Ms,F)
plt.plot(Ms,FRay)
plt.ylim([-1.9,3.5])
plt.grid()
plt.xscale('log')
plt.show()

np.save('Forces_mie.npy',F)
np.save('Masses_mie.npy',Ms)