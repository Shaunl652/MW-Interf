# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 13:07:24 2021

@author: shaun
"""
from scipy.constants import atomic_mass
from scipy.special import erf
from numpy import pi, sqrt

m0 = atomic_mass

Gamma = lambda lambda_CSL, r_C, expt: \
    (expt.mass/m0)**2 * lambda_CSL

f = lambda x,r_C,expt: \
    sqrt(pi)*r_C/x * erf(x/(2*r_C))
