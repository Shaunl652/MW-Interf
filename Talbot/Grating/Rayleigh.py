import numpy as np
from numpy import pi, cos, sin, exp

sgn = np.sign
Re = np.real
Im = np.imag
from scipy.special import jn

def func(n,u,v):
    # =========================================================================
    # We can take the real part here as analytically any imaginary numbers
    # Cancel out from the fraction and the bessel function 
    # ==========================================================================
    from scipy.special import jn
    import numpy as np
    sign = np.sign
    from numpy.lib.scimath import sqrt
    return (sqrt(u/v)**n*jn(n,sign(v)*sqrt(u*v))).real

def B_n(n,expt):
    eta = n*expt.eta
    chi  = expt.chi
    beta = Im(chi)/Re(chi)
    Zcoh = expt.phi0*sin(pi*eta)
    if expt.grating_photon_absorption:
        Zabs = expt.phi0*beta*(1-cos(pi*eta))
    else:
        Zabs = 0

    return exp(-Zabs)*func(n,Zcoh+Zabs,Zcoh-Zabs)
