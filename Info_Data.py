# ===============================================================================
# Finds the expected information and Lambda as a function of data points 
# When called take the scenario name, and prior type
# ===============================================================================

import numpy as np
import Functions.IntFit as IntFit
from Talbot.CSL.Extended import lambc
from Talbot.Likelihood import Like
from random import choices
import Bayesian.Prior as pri
import json
import sys
from Experiment import Experiment
import time
from multiprocessing import Pool
from functools import partial
from pathlib import Path
from Optomise import OptFunc
import Bayesian.Info as Info


prm_file = "Scenarios/"+sys.argv[1]+".json"

# Makes the folder to store the mass dependant info and Lambda estimate
npy_path = f"NPYs/{sys.argv[1]}/{sys.argv[2]}/Data"
Path(npy_path).mkdir(parents=True, exist_ok=True) 


with open("Space.json","r") as read_file:
    Spc = json.load(read_file)

lam_axis = np.logspace(Spc['Lambda']['Min'],Spc['Lambda']['Max'],Spc['Lambda']['Step'])
r_C_axis = np.logspace(Spc['r_c_ps']['Min'],Spc['r_c_ps']['Max'],Spc['r_c_ps']['Step'])




#nVals = np.logspace(0,4.2,501)
#Ns = sorted(list(set([int(n) for n in nVals])))
Ns = [*range(0,16000,200)]

expt = Experiment(sys.argv[1])
expt.phi0,expt.a2 = OptFunc(expt)
xaxis    = expt.xaxis
print("Finding the simulation weightings") 
prob = Like(expt,CSL=None) # Just chooses x based on the likelihood where theta=0

# Likelihood for w(x|theta)
print("Finding the liklihood")
lkhd = Like(expt)


# Finds the Prior based on given comands
if sys.argv[2] == "Jeff":
    Prior = pri.Jeff(lkhd)
elif sys.argv[2] == "Experimental":
    Prior = pri.Experimental()
elif sys.argv[2] == "MDIP":
    Prior = pri.MDIP(lkhd,xaxis)
else:
    sys.exit('Unrecognised prior type')
Prior /= np.trapz(np.trapz(Prior,lam_axis,axis=0),r_C_axis)
logJ = np.log10(Prior)
lkhd_dict = {x: lkhd[i] for i,x in enumerate(expt.xaxis)}

EH = []
Er = []
M = 200
for n in Ns:
    print(n)
    start_time = time.time()
    if sys.argv[3] == 'Theta':
        EHLoop = []
        for m in range(M):
            x_data = choices(expt.xaxis,weights=prob,k=n)
            # Finds the posterior from the data x_data
            logLikelihood = sum((np.log10(lkhd_dict[x]) for x in x_data))
            # Introduces a reduction term to avoid numerical errors, is canceld out in the normalisation
            kappa = np.max(logLikelihood)
            logLikelihood = logLikelihood - kappa
            # Finds the posterioir
            Post = np.exp(logJ + logLikelihood)
            # Normalises the posterior
            Evid = np.trapz(np.trapz(Post, r_C_axis,axis=1),lam_axis)
            Post /= Evid
    


            # Get the information
            EHLoop.append(Info.H(Post,Prior))
        Er.append(np.var(EHLoop))
        EH.append(np.mean(EHLoop))
    else:
        Info_Out,var  = Info.Utility(lkhd, Prior, expt.xaxis)
        EH.append(Info_Out) # Saves the expected info values
        Er.append(var)
    
    print(f'Finished loop in {(time.time()-start_time)/60:.2f} mins')

take = sys.argv[3]
np.save(f"{npy_path}/Info{take}.npy",np.array(EH))
np.save(f"{npy_path}/Err_{take}.npy",np.array(Er))





















