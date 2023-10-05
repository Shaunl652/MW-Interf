# ====================================================================================
# Finds the prior, can either do Jeff prior or flat prior with non-interferometric tests
# ====================================================================================

import numpy as np
from numpy import exp, sqrt, trapz
import json
diff = np.gradient


with open("Space.json","r") as read_file:
    Spc = json.load(read_file)


lam_axis = np.logspace(Spc['Lambda']['Min'],Spc['Lambda']['Max'],Spc['Lambda']['Step'])
r_C_axis = np.logspace(Spc['r_c_ps']['Min'],Spc['r_c_ps']['Max'],Spc['r_c_ps']['Step'])



# Finds the Prior
# ===============================================================================================================================

def loc(a,N):
    """Finds the location of N or the first number larger than N in sorted array a"""
    n = a[0] # Checks the first location in the array
    i = 0    # Initialises the counter variable
    # Checks if the value at the ith position in the array is less than N, if so repeat
    while n < N: 
        i += 1   # Incriments counter
        n = a[i] # Sets the new value to check
    return i

# Find the values on the non-interferometric exclusion lines that extend to the edges of the plot incase we extend the parameter space
# Calculated these based on the plots in Fig2 in [Gasbarri PRA (2021)]

def Non_Int_Line(x):
    """Finds the value of lambda at r_C (x) to bound the prior by the non-iterferometric upper bounds"""
    if x <= 7e-7:
        return 1835*x**2.126
    elif x > 7e-7 and x <= 2e-2:
        return 1e-21 * x**(-1.845)
    else:
        return 1e-14 * x**2.38

# Finds the Jefferys' Prior based on equations given in :
# [Robert 'Harold Jeffreys's theory of probability revisited' Stat Sci 24 (2009)]
def Jeff(lkhd,xaxis):
    dwdl = diff(np.log(lkhd),lam_axis,axis=1)
    dwdr = diff(np.log(lkhd),r_C_axis,axis=2)
    
    Fish_11 = trapz(dwdl**2   *lkhd, xaxis, axis=0)
    Fish_22 = trapz(dwdr**2   *lkhd, xaxis, axis=0)
    Fish_12 = trapz(dwdl*dwdr *lkhd, xaxis, axis=0)

    # Trying this method, See Peter Lee section 3.3 for more details
    DetFish = Fish_11 * Fish_22 #- Fish_12**2 

    return sqrt(DetFish) + 1e-99

# Builds the prior based on experimental results
# Based on the exclusion plot presented in [Gasbarri PRA 103 (2021)]
def Experimental():
    
    PSpace = np.zeros((Spc['Lambda']['Step'],Spc['r_c_ps']['Step'])) # Create an array of 0s

    for j in range(PSpace.shape[1]):
        L = Non_Int_Line(r_C_axis[j])
        for i in range(PSpace.shape[0]):
            if lam_axis[i] <= L:
                PSpace[i][j] = 1

    return PSpace + 1e-99

def Flat():
    PSpace = np.ones((Spc['Lambda']['Step'],Spc['r_c_ps']['Step'])) # Create an array of 0s
    return PSpace + 1e-99

def MDIP(lkhd,xaxis):
    # Derives the maximal data information prior as given in Yang & Berger (1998) pg6
    return exp(trapz(lkhd*np.log(lkhd),xaxis,axis=0))
    
