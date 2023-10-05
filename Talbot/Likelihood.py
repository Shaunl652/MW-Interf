# =============================================================================
# Finds the likelihood function w(x|theta)
# =============================================================================
import numpy as np
from numpy import exp
from numpy.lib.scimath import sqrt
from scipy.special import gamma, sici,spherical_jn
Si = lambda x: sici(x)[0]
sinc = lambda x: np.sinc(x/np.pi)
J1 = lambda x: spherical_jn(1, x)
import json


# Constants
# ===========================================================================================================================
import scipy.constants as const
hbar = const.hbar
kB = const.k
h = const.Planck
pi = np.pi 
epsilon_0 = const.epsilon_0
AMU = const.atomic_mass
c = const.speed_of_light
diff = np.gradient

Re = np.real
Im = np.imag
sin = np.sin
cos = np.cos
from scipy.special import jn
sgn = np.sign

# Parameter space
# ===========================================================================================================================
with open("Space.json","r") as read_file:
    Spc = json.load(read_file)

lam_axis = np.logspace(Spc['Lambda']['Min'],Spc['Lambda']['Max'],Spc['Lambda']['Step'])
r_C_axis = np.logspace(Spc['r_c_ps']['Min'],Spc['r_c_ps']['Max'],Spc['r_c_ps']['Step'])


# Functions
# ===========================================================================================================================
def Like(expt,Talbot="Mie",Decoherence=True,Therm=True,CSL="Extended",L=None,R=None):
    """
    Finds the likelihod of measuring the particle at x for a given theta

    Parameters
    ----------
    expt : class
        Experimental parameters
    Talbot : optional
        Specify the type of Talbot coefficients: "Rayleigh" or "Mie".  Default is "Mie".
    Decoherence : optional, boolean
        Specify whether to include decoherence. Default is True
    CSL : optional
        Specify the type of CSL: None, "Pointlike", "Extended".  Default is "Extended".

    Returns
    -------
    outputs : Array of Floats
        An array describing the likelihood of measuring the particle for the given parameter space.

    """
    N = 6                   # 6 is the best, any larger and the cosines are pretty much all 0 for all x
    Ns = [*range(N+1)]      # List of integers from 0 to N

    # Talbot coefficients
    # =======================================================================================================================
    from Talbot.Grating.Coefficients import Coef
    T_dict = Coef(N,expt,Talbot=Talbot,Therm=Therm)

    # Conventional Decoherence sources
    # =======================================================================================================================
    if Decoherence:
        from Talbot.Decoherence import Deco
        lnR_dec = Deco(N,expt)
    else:
        lnR_dec = {n: 0 for n in Ns}

    # CSL effect
    # =======================================================================================================================
    R,L = np.meshgrid(r_C_axis,lam_axis)
    from Talbot.CSL.Coefficients import lnR_CSL_func
    lnR_CSL = lnR_CSL_func(N,L,R,expt,CSL=CSL)
    
    #return T_dict, lnR_dec, lnR_CSL, Ns, xaxis

    # Build w(x|theta)
    # =======================================================================================================================
    outputs = np.array([1+2*sum([T_dict[n]*exp(lnR_dec[n])*exp(lnR_CSL[n])* np.cos(n*2*pi*x/expt.D) for n in Ns[1:]]) for x in expt.xaxis]) * expt.mass/(sqrt(2*pi)*expt.sigmap*(expt.t1+expt.t2)) # list-comp
    #outputs = (1+2*sum([T_dict[n]*exp(lnR_dec[n]+lnR_CSL[n][None,:,:]) * np.cos(n*2*pi*xaxis[:,None,None]/expt.D) for n in Ns[1:]])) * expt.mass/(sqrt(2*pi)*expt.sigmap*(expt.t1+expt.t2))   # numpy broadcasting (marginally faster ~2%)
    return outputs

