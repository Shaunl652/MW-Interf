#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
# This code is the function to do the bayesian updating
# =============================================================================


import numpy as np
from numpy import trapz, log2, exp
import json
from random import choices
from time import time
    
with open("Space.json","r") as read_file:
    Spc = json.load(read_file)

lam_axis = np.logspace(Spc['Lambda']['Min'],Spc['Lambda']['Max'],Spc['Lambda']['Step'])
r_C_axis = np.logspace(Spc['r_c_ps']['Min'],Spc['r_c_ps']['Max'],Spc['r_c_ps']['Step'])

def Bayes_Analysis(X_data,Prior,lkhd):
    """
    Finds the posterior from the prior, likelihood and the X data 

    Parameters
    ----------
    X_data : ARRAY
        An array of all the measured x locations.
    Prior : ARRAY
        The normalised prior.
    lkhd : DICTIONARY
        A dictionary of the likelihood, indicies relate to the possible x locations.

    Returns
    -------
    Post : ARRAY
        The final normalised posterior P(theta|X).

    """
    
    logLikelihood = sum((np.log(lkhd[x]) for x in X_data))
    # Introduces a reduction term to avoid numerical errors, is canceld out in the normalisation
    kappa = np.max(logLikelihood)
    logLikelihood = logLikelihood - kappa
    # Finds the posterioir
    Post = np.exp(np.log(Prior) + logLikelihood)
    # Normalises the posterior
    Evid = trapz(trapz(Post, r_C_axis,axis=1),lam_axis)
    Post /= Evid
    
    return Post