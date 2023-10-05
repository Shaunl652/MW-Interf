# =============================================================================
# Deals with the Talbot Coefficents, Based on the equations presented in 
# [Belechia, 'Talbot-Lau effect beyond the point-particle approximation', PRA 100, (2019)]
# But we convert them to work with scattering amps from Scuff-EM (e.g. for hexagonal plates)
# ============================================================================


# General Imports, basic functions, and constants
# ============================================================================
import numpy as np

import Talbot.MieScatter as ms
from scipy.interpolate import griddata

from scipy.special import jv
Re = np.real
Im = np.imag
conj = np.conj
cos = np.cos
sin = np.sin
exp = np.exp
sgn = np.sign
import scipy.constants as const
pi = np.pi
c = const.speed_of_light
hbar = const.hbar
h = const.Planck
epsilon_0 = const.epsilon_0




# Spherical integration library
# ============================================================================
import quadpy
scheme = quadpy.u3.get_good_scheme(47)

# Coherent and Absorption Effects from the Grating
# ============================================================================
def zeta_cohf(eta,phi0):
    return phi0 * sin(pi*eta)

def zeta_absf(eta,n_0):
    return n_0*(1-cos(pi*eta))/2

# Finds the Talbot Coefficents
# ============================================================================

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


# Finds the conovlution kernal for each incoherent scattering effect
# =============================================================================

def Scatter_funcs(n,fs,F0,expt):
    s = n*expt.s
    k = expt.k
    # Pull out each of the scattering amplitudes
    fx_pos,fy_pos,fz_pos,fx_neg,fy_neg,fz_neg = fs
    # From the force we compute the fluence which gives rise to the specified phase
    fluence = expt.phi0*hbar*c*expt.k**3/(4*F0)
    
    def abFIntegrand(theta_phi):
        theta,phi = theta_phi

        axIntegrand = Re(fx_pos.conj()*fx_neg)*(cos(k*cos(theta)*s)-cos(k*s))
        bxIntegrand = Im(fx_pos.conj()*fx_neg)* sin(k*cos(theta)*s)
        FxIntegrand = abs(fx_pos)**2*(cos((1-cos(theta))*k*s)-1)
        
        ayIntegrand = Re(fy_pos.conj()*fy_neg)*(cos(k*cos(theta)*s)-cos(k*s))
        byIntegrand = Im(fy_pos.conj()*fy_neg)* sin(k*cos(theta)*s)
        FyIntegrand = abs(fy_pos)**2*(cos((1-cos(theta))*k*s)-1)
        
        azIntegrand = Re(fz_pos.conj()*fz_neg)*(cos(k*cos(theta)*s)-cos(k*s))
        bzIntegrand = Im(fz_pos.conj()*fz_neg)* sin(k*cos(theta)*s)
        FzIntegrand = abs(fz_pos)**2*(cos((1-cos(theta))*k*s)-1)
        return np.array([axIntegrand, bxIntegrand, FxIntegrand, ayIntegrand, byIntegrand, FyIntegrand, azIntegrand, bzIntegrand, FzIntegrand])
    
    # Integrate the a, b, and F values
    # ====================================================================================
    coef = abs(8*pi*fluence/(hbar*c*expt.k)) # Take the abs as a -ve fluance doesn't make physical sense (result of -ve force)
    ax,bx,Fx,ay,by,Fy,az,bz,Fz = coef * scheme.integrate_spherical(abFIntegrand)
    
    return ax,bx,Fx,ay,by,Fy,az,bz,Fz



def B_n(n,expt,fs):
    """
    Finds the Talbot Coefficients based on the equations in [Belenchia et al. PRA 100 (2019)]
    This one is called if the scattering terms have not been generated

    Parameters
    ----------
    n : Integer
        The order of B_n to calculate.
    expt : class
        Experimental parameters

    Returns
    -------
    Float
        Value for B_n.

    """
    
    # Summation over k, -10 to 10 should be enough values
    krange = 10
    kvals = np.arange(-krange,krange+1)

    # parameter passed to the B_n funtion (given as s/d in Belenchia)
    eta = n*expt.eta

    # Calculating the z component of the force, Using the unitless form and rearanging later equations to use this form
    from Talbot.MieForce import F_z
    F0 = abs(F_z(-expt.lambdaG/8,expt.k,expt.Radius,expt.RefInd)) # We want the MAGNITUDE of the force

    # Cross sections are given in nm^2, convert to m^2 by multiplying by 1e-18, also takes in WL and R in nm
    CrossSections = ms.MieQ(expt.RefInd,expt.lambdaG/1e-9,expt.Radius*2/1e-9,asDict=True,asCrossSection=True)

    # Mean number of absorbed photons, given just under EQ(37)
    if expt.grating_photon_absorption == False:
        n_0 = 0
    else:
        n_0 = (expt.k**2/F0)*CrossSections['Cabs']*1e-18 * expt.phi0

    # Finds the parallel and perpendicular forms of the scattering functions
    ax,bx,Fx,ay,by,Fy,az,bz,Fz = Scatter_funcs(n,fs,F0,expt)
    
    # Finds the total scattering terms
    a_tot = ax + ay + az
    b_tot = bx + by + bz
    F_tot = Fx + Fy + Fz

    # Gets the absorption and coherent effects
    z_abs = zeta_absf(eta, n_0)
    z_coh = zeta_cohf(eta, expt.phi0)
    
    # Putting everthing togther   
    PreFact = exp(F_tot - z_abs)
    func_vals = np.array([func(n+k,z_coh+a_tot+z_abs,z_coh-a_tot-z_abs) for k in kvals])
    Bessel = np.array([jv(k,b_tot) for k in kvals])
    return PreFact * sum(Bessel*func_vals)

 

