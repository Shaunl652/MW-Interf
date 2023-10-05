# -*- coding: utf-8 -*-
# ============================================================================= 
# Code to find the optimum parameters for a given experimnt
# The aim of this code is to find the optimum parameters of t2 and phi0 by 
# finding the pair of parameters that leads to the maximum change to the 
# visibility as a result of the CSL modification
# This code takes 2 arguments: 
#     1) scenario file to optomise (pass the name only e.g. "Example" not "Scenarios/Example.json")
#     2) Scenario file to write the code to (given the same as above)
#        If no output file is given, we overwrite the input file
# =============================================================================

import numpy as np
from Experiment import Experiment
from sys import argv
import json
from Talbot.Grating.Mie import B_n
from Talbot.Grating.Coefficients import Coef
from Talbot.CSL.Coefficients import lnR_CSL_func
from scipy.constants import atomic_mass as AMU
from scipy.interpolate import interp2d

CSLVals = np.load('NPYs/CSLVis.npy')
InterpX = np.logspace(6,10,51)*AMU # masses used in interpolation data
InterpY = np.linspace(0.01,3,51) # times used in interpolation data

CSLInterp = interp2d(InterpX,InterpY,CSLVals)


def OptFunc(expt,idx_plot=0):
    
    phi_axis = np.linspace(0.1*np.pi,2*np.pi,100)
    t_2_axis = np.linspace(0.5,1.5,100)
    
    Vis_vals = [[] for i in phi_axis]
    Vis_mods = [[] for i in phi_axis] # Contains the CSL reduced visability
    for idx,phi in enumerate(phi_axis):
        expt.phi0 = phi
        # print(f'\n{idx:03}/{len(phi_axis)} : {phi/np.pi:.2f}pi')
        for jdx,t2 in enumerate(t_2_axis):
            # print(f'\r{jdx:03}/{len(t_2_axis)} : {t2:.2f}t2/tT',end='')
            expt.a2 = t2
            vals = Coef(1,expt)
            # This is the visibility for the given phi0 and t2
            Vis_idx = 2*abs(vals[1])*expt.mass/(np.sqrt(2*np.pi)*expt.sigmap*(expt.t1+expt.t2))
            # Store the current visibility
            Vis_vals[idx].append(Vis_idx)
            # We choose these values of the CSL parameters to match the lowest point on the minimum eclusion line
            # If the line is lower than this we rule out CSL compleatly
            CSL = np.exp(lnR_CSL_func(1, 1e-20, 1e-5, expt)[1])
            #CSL = CSLInterp(expt.mass,expt.a2)[0]
            
            Vis_mods[idx].append(Vis_idx*CSL)
            
    # Make these both arrays so that we can do the maths with them
    Vis_vals = np.array(Vis_vals).T
    Vis_mods = np.array(Vis_mods).T
    
    # Finds the difference between the no CSL plot and the CSL modified plot    
    reduction = Vis_vals - Vis_mods
    
    
    # Finds the optimum values and stores them in the working_file dict
    max_idx = np.unravel_index(reduction.argmax(), Vis_vals.shape)
    

    # import matplotlib.pyplot as plt
    
    # plt.pcolormesh(phi_axis,t_2_axis,reduction)
    # plt.colorbar()
    # plt.scatter(phi_axis[max_idx[1]],t_2_axis[max_idx[0]])
    # plt.ylabel('$t_2/t_T$')
    # plt.xlabel('$\\phi_0$')
    # plt.title(f'Mass = {expt.mass/AMU:.3e} AMU')
    # plt.savefig(f'Plots/Vis/{idx_plot+1}_{expt.mass/AMU:.3e}'.replace('.',',')+'.png')
    # plt.close()

    return phi_axis[max_idx[1]], t_2_axis[max_idx[0]]

if __name__ == '__main__':

    # Defines the experimental parameters
    expt = Experiment(argv[1])

    print('experiment')
    with open(f'Scenarios/{argv[1]}.json',"r") as read_file:
            working_file = json.load(read_file)
    
    phi_origional = working_file['Phi0']
    t_2_origional = working_file['a2']
    

    print('Working')
    working_file['Phi0'],working_file['a2'] = OptFunc(expt)

    if len(argv) >= 3:
        output = argv[2]
    else:
        yn = input('This will overwrite the Scenario file. Are you sure? (y/n)\n')
        while True:
            if yn == 'y':
                output = argv[1]
                break
            elif yn =='n':
                output = input('Please enter name for the new scenario file:\n')
                break
            else:
                yn = input('Input not recognised, please type either "y" or "n"\n')
     
            
    print('Origional values:')
    print(f'Phi0 = {phi_origional}')
    print(f't2/tT = {t_2_origional}')
    
    print('Optomised Values:')
    print(f"Phi0 = {working_file['Phi0']}")
    print(f"t2/tT = {working_file['a2']}")
    
    
    with open(f'Scenarios/{output}.json','w') as out_file:
        json.dump(working_file,out_file,indent=3)
    

    


