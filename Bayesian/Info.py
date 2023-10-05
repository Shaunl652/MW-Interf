
"""
Functions to find information
"""

import numpy as np
from numpy import trapz, log2, exp
import json
from random import choices
from time import time
    
with open("Space.json","r") as read_file:
    Spc = json.load(read_file)

lam_axis = np.logspace(Spc['Lambda']['Min'],Spc['Lambda']['Max'],Spc['Lambda']['Step'])
r_C_axis = np.logspace(Spc['r_c_ps']['Min'],Spc['r_c_ps']['Max'],Spc['r_c_ps']['Step'])


def H(Post,Prior):
    """
    Finds the information gain between the normalied prior and posterior

    Parameters
    ----------
    Post : ARRAY
        The normalised posterior.
    Prior : ARRAY
        The normaised prior.

    Returns
    -------
    FLOAT
        The information gain in the posterior from the prior.

    """

    Inte = Post * np.nan_to_num(np.log2(Post/Prior)) # finds the integrand
    return trapz(trapz(Inte, lam_axis,axis=0),r_C_axis)

    
def log_Evid(Xs,lkhd,Prior,xaxis):
    """
    Finds the log of the marginal probability of the vector Xs

    Parameters
    ----------
    Xs : ARRAY
        The vector of measured x locations.
    lkhd : ARRAY
        The conditional likelihood P(x|Theta), must be normalised wrt x.
    Prior : ARRAY
        The prior distribution P(Theta), must be normalised.
    xaxis : ARRAY
        An array of all possible measurement locations x.

    Returns
    -------
    ARRAY
        Marginal probability of measuring the vector X, P(X).

    """
    
    
    # First find the log of the lkhd and store in a dict of x loca
    lkhd_log_dict = {x: np.log2(lkhd[i]) for i,x in enumerate(xaxis)}
    loglkhd = sum([lkhd_log_dict[x] for x in Xs]) # find the log of the lkhd for a given X vector
    kappa = loglkhd.max() # Kappa is a scaling factor to avoid numerical issues later
    loglkhd = loglkhd - kappa
    post = 2**(loglkhd + Prior) # The un-noralised posterior P(theta|X)
    return np.log2(np.trapz(np.trapz(post,lam_axis,axis = 0),r_C_axis)) + kappa

def Utility(lkhd,Prir,xaxis,N=10000,M=1000):
    
    # Time stuff for testing
    from time import time
    start_time = time()
    
    # To find the evidance we want the normalised lkhd and prior, but elsewhere we want the un-normed versions
    lkhd_norm = lkhd/np.trapz(lkhd,xaxis,axis=0)
    Prir_norm = Prir/np.trapz(np.trapz(Prir,lam_axis,axis = 0),r_C_axis)
    
    # This block will find the coords of theta picked form the prior in terms of array elements
    flat_Prior = Prir_norm.flatten() # Flatten the prior to work with choices
    theta_coords = [(l,r) for l in range(len(lam_axis)) for r in range(len(r_C_axis))] # list of lambda_C,r_C coords wrt flattened prior
    theta_i_vals = choices(theta_coords,weights=flat_Prior,k=M) # picks the values of the theta coords
    
    # Puts the lkhd into a dict of x for faster accessing

    #lkhd_dict = {x: lkhd_norm[i] for i,x in enumerate(xaxis)}
    
    
    Infos = [] # Stores the value of the info (interior of the sum)
    Infos_square = [] # Stores the square of the info to calculate the variance
    times = []
    for n,t_i in enumerate(theta_i_vals):
        print(f'\rWorking on n={n+1} : theta={t_i}        ',end='')
        loop_start_time = time()
        
        # P(x|theta_i) used to pick the values of X^i = [x_1^i,x_2^i,...x_M^i] 
        prob_X = lkhd_norm[:,t_i[0],t_i[1]] 
        
        # Convert p(x|theta_i) into a dict for faster accessing
        prob_x_dict = {x: prob_X[i] for i,x in enumerate(xaxis)} 
        
        # Builds the vector X^i (described above)
        X_vec = choices(xaxis,prob_X,k=N) # This is allowed as p(X|THETA) = prod_i p(x_i|THETA)
        
        # Finds the value of  the log of P(X_i), we want the log anyway so leave it as it is
        Px = log_Evid(X_vec,lkhd_norm,np.log2(Prir_norm),xaxis)
        # Find the likelihood of the vector P(X^i|Theta)
        # Want log of p(X^ii|theta_i) anyway, doing the sum is better for numerical stability
        Px_give_theta = sum([np.log2(prob_x_dict[x]) for x in X_vec]) 
        
        # We don't include logs here as these values have already had the log taken
        sum_int = Px_give_theta - Px # Sum int is the interior of the sum (Sum_i (sum_int))
        Infos.append(sum_int)
        Infos_square.append(sum_int**2) # Square for variance
        # Output loop time
        times.append(time() - loop_start_time)
        
    value = sum(Infos)/M # This is the final expected information
    std = np.sqrt(np.var(Infos)/M) # This is the variance in the expected information
    
    print(f'Total time = {(time()-start_time)/60:.4f} mins')
    print(f'Ave time per loop = {sum(times)/M:.4f} +/- {np.std(times):.4f} seconds')
    return value, std
    
    
    
    
    
    
    
    
    