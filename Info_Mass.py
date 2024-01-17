# =============================================================================
# Finds the expected information and Lambda as a function of Mass for MCMC
# Reads in the scenario name, and the type of prior
# =============================================================================


import numpy as np
from Talbot.Likelihood import Like
import Bayesian.Prior as pri
import json
import sys
from Experiment import Experiment
import time
from pathlib import Path
from Optomise import OptFunc
from Bayesian.Info import Utility


import scipy.constants as const
hbar = const.hbar
kB = const.k
h = const.Planck
pi = np.pi 
epsilon_0 = const.epsilon_0
AMU = const.atomic_mass
c = const.speed_of_light
diff = np.gradient
trapz = np.trapz


# Start time
Init_Time = time.time()


# Makes the folder to store the mass dependant info and Lambda estimate
npy_path = f"NPYs/{sys.argv[1]}/{sys.argv[2]}/Mass"
Path(npy_path).mkdir(parents=True, exist_ok=True) 


with open("Space.json","r") as read_file:
    Spc = json.load(read_file)

lam_axis = np.logspace(Spc['Lambda']['Min'],Spc['Lambda']['Max'],Spc['Lambda']['Step'])
r_C_axis = np.logspace(Spc['r_c_ps']['Min'],Spc['r_c_ps']['Max'],Spc['r_c_ps']['Step'])

Ms = np.logspace(6,10,100)
expt = Experiment(sys.argv[1]) # Calculates the experimental variables

    

Info_Vals = []
Info_Vars = []

for m in Ms:
    start_time = time.time()
    print(f"Starting mass {m:.2e}")
    expt.mass = m*AMU
    # Optomises the parameters
    expt.phi0,expt.a2 = OptFunc(expt)
    print(f'Optomisation finished at t = {time.time()-start_time:.2f} s')
    xaxis = expt.xaxis
    lkhd = Like(expt)
    prob = Like(expt,CSL=None) # Just chooses x based on the likelihood where theta=0
    # Builds the prior for each new experiment
    if sys.argv[2] == "Jeff":
        Prior = pri.Jeff(lkhd)
    elif sys.argv[2] == "Experimental":
        Prior = pri.Experimental()
    elif sys.argv[2] == "MDIP":
        Prior = pri.MDIP(lkhd,xaxis)
    elif sys.argv[2] == "Flat":
        Prior = pri.Flat()
    else:
        sys.exit('Unrecognised prior type')
    

    # Get the information from the MCMC if we are not plotting the 
    Info_Out,var  = Utility(lkhd, Prior, expt.xaxis)
    Info_Vals.append(Info_Out) # Saves the expected info values
    Info_Vars.append(var) # Saves the variance values
    
    # Now we need to see the values of lambda_c
    Prior /= np.trapz(np.trapz(Prior,lam_axis,axis=0),r_C_axis)
    print(f'Found prior at t = {time.time()-start_time:.2f} s')
    logJ = np.log10(Prior)
    lkhd_dict = {x: lkhd[i] for i,x in enumerate(expt.xaxis)}
   
    
    print(f'Finished loop in {(time.time()-start_time)/60:.2f} mins')
    
np.save(f"{npy_path}/Info_vals_MCMC.npy",np.array(Info_Vals))
np.save(f"{npy_path}/Info_Vars_MCMC.npy",np.array(Info_Vars))
np.save(f"{npy_path}/Mass.npy",Ms)































