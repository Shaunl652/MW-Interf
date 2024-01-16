# -*- coding: utf-8 -*-

# =============================================================================
# This is the code to find the expected minimum lambda_c upper bound for a constant
# r_c = 1e-7m as a function of mass
# This will produce data for the Experimental prior and the MDIP as seen in Fig.5
# Also finds the expected info should theta=0 (Fig.4(b))
# Reads in {Scenario Name} and {Prior Type}
# =============================================================================


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
from Functions.find_phi0 import Opt_Phi
from Optomise import OptFunc
from random import choices
from Bayesian.Info import Utility, H


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
npy_path = f"NPYs/{sys.argv[1]}/{sys.argv[2]}/Mass/Theta0"
Path(npy_path).mkdir(parents=True, exist_ok=True) 


with open("Space.json","r") as read_file:
    Spc = json.load(read_file)

lam_axis = np.logspace(Spc['Lambda']['Min'],Spc['Lambda']['Max'],Spc['Lambda']['Step'])
r_C_axis = np.logspace(Spc['r_c_ps']['Min'],Spc['r_c_ps']['Max'],Spc['r_c_ps']['Step'])

Ms = np.logspace(6,10,100)
expt = Experiment(sys.argv[1]) # Calculates the experimental variables

    

Info_Vals = []
Info_Vars = []
lamb_vals = []
lamb_vars = []
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
    
    
    # Now we need to see the values of lambda_c
    Prior /= np.trapz(np.trapz(Prior,lam_axis,axis=0),r_C_axis)
    print(f'Found prior at t = {time.time()-start_time:.2f} s')
    logJ = np.log10(Prior)
    lkhd_dict = {x: lkhd[i] for i,x in enumerate(expt.xaxis)}
    # Random x data
    lambda_mass = [] # Stores the values of lambda_c for this mass
    EHLoop = []
    for i in range(1000):
        print('Finding lambda_c')
        print(f'\rRunning loop {i:04}/1000')
        x_data = choices(expt.xaxis,weights=prob,k=10000)
        # Finds the posterior from the data x_data
        logLikelihood = sum((np.log10(lkhd_dict[x]) for x in x_data))
        # Introduces a reduction term to avoid numerical errors, is canceld out in the normalisation
        kappa = np.max(logLikelihood)
        logLikelihood = logLikelihood - kappa
        # Finds the posterioir
        Post = np.exp(logJ + logLikelihood)
        # Normalises the posterior
        Evid = trapz(trapz(Post, r_C_axis,axis=1),lam_axis)
        Post /= Evid
        
        Lambda, Conf = IntFit.FindLambda(Post, 1e13, expt,err=0.001)
        
        # Find lambda_c at r_c=1e-7
        lambda_mass.append(lambc(1e-7,Lambda,expt))
        
            
        EHLoop.append(H(Post,Prior))
    
    # Stores the values of the mean infos and lambda_c and the variances
    Info_Vals.append(np.mean(EHLoop))
    Info_Vars.append(np.var(EHLoop))
        
    lamb_vals.append(np.mean(lambda_mass))
    lamb_vars.append(np.var(lambda_mass))
    
    print(f'Finished loop in {(time.time()-start_time)/60:.2f} mins')
    
np.save(f"{npy_path}/Info_vals.npy",np.array(Info_Vals))
np.save(f"{npy_path}/Info_Vars.npy",np.array(Info_Vars))
np.save(f"{npy_path}/Lamb_Vals.npy",np.array(lamb_vals))
np.save(f"{npy_path}/Lamb_Vars.npy",np.array(lamb_vars))
np.save(f"{npy_path}/Mass.npy",Ms)
