# =============================================================================
# Plots the differnet sources of decoherence as a function of mass
# Uses a scenario given when calling the code
# =============================================================================

import numpy as np
from Talbot.Decoherence import Deco
from Experiment import Experiment
import sys
from Functions.find_phi0 import Opt_Phi
from Optomise import OptFunc
import scipy.constants as const
import json
from pathlib import Path

AMU = const.atomic_mass


#expt = Experiment(sys.argv[1])

Ms = np.logspace(6,10,101)
Total = False
NPYPath = f'NPYs/Decoherence/{sys.argv[1]}'
Path(NPYPath).mkdir(parents=True,exist_ok=True)

# if Total:
#     vals = []
#     for idx,m in enumerate(Ms):
#         print(f'\r{idx:03}/{len(Ms)} : {m:.2e} AMU',end='')
#         expt.mass = m*AMU
#         expt.phi0,expt.a2 = OptFunc(expt)
        
#         lnRs = Deco(1,expt) 
        
#         vals.append(np.exp(lnRs[1]))
#     np.save(f'{NPYPath}/{sys.argv[2]}.npy',np.array(vals))
# else:
#     sources = ['Collision','Absorption','Scattering','Emission','Uncert']
#     Deco_Dict = {s: [] for s in sources}
    
#     for idx,m in enumerate(Ms):
#         print(f'\r{idx:03}/{len(Ms)} : {m:.2e} AMU',end='')
#         expt.mass = m*AMU
#         expt.phi0,expt.a2 = OptFunc(expt)
        
#         lnRs = Deco(1,expt,Seperate=True)
        
#         for jdx,s in enumerate(sources):
#             Deco_Dict[s].append(np.exp(lnRs[jdx][1]))
            
#             with open(f'DecoData/{sys.argv[1]}.json','w') as f:
#                 json.dump(Deco_Dict,f)
    
import matplotlib.pyplot as plt
with open(f'DecoData/{sys.argv[1]}.json','r') as f:
    Deco_Dict = json.load(f) 
sources = ['Collision','Absorption','Scattering','Emission','Uncert']
labels = ['Collision','Absorption','Scattering','Emission','Position Uncertainty']
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set(xlabel='Mass [u]',xscale='log',ylabel='Visibility Modification')
for i,s in enumerate(sources):
    ax.plot(Ms,Deco_Dict[s],label=labels[i])
    
ax.grid(which='both')
ax.legend()
fig.show()
    
    
    
