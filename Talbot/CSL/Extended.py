# ==============================================================================
# Contains all the functions used to model the effects of CSL
# ==============================================================================

from numpy import pi, sqrt, exp, array
from scipy.constants import atomic_mass, Planck
from scipy.special import spherical_jn, sici

# USeful functions
# ===========================================================================================================================
Si = lambda x: sici(x)[0]
J1 = lambda x: spherical_jn(1, x)
def integrate(func, a, b):
    from numpy import linspace, array, trapz
    xs = linspace(a,b,1001)
    vals = array([func(xi) for xi in xs])
    return trapz(vals,xs,axis = 0)

# Useful values
# ===========================================================================================================================
integral_limits = 1e-20,10 # exp(-a**2) pulls integrand towards zero very quickly; no point integrating above ~100
h  = Planck
m0 = atomic_mass

# CSL rate and spatial resolution parameters, based on equations from [Gasbarri Communications Physics 4 (2021)]
# =============================================================================================================================
# Finds Gamma (rate parameter)
Gamma = lambda lambda_CSL, r_C, expt: \
    36/sqrt(pi)*(expt.mass/m0)**2*(r_C/expt.Radius)**2*lambda_CSL * \
    integrate(lambda a: exp(-a**2)*J1(a*expt.Radius/r_C)**2,*integral_limits)

# Finds f(x)  (spatial resolution)
f = lambda x,r_C,expt: \
    (r_C/x) * integrate(lambda a: exp(-a**2)*J1(a*expt.Radius/r_C)**2*Si(a*x/r_C)/a,*integral_limits) / \
    integrate(lambda a: exp(-a**2)*J1(a*expt.Radius/r_C)**2,*integral_limits)


def fcsl_size(x):
    
    if x <= 0.005:
        return 1
    elif x == 1:
        return 0.62
    else:
        prefac = 6/(x**4)    
        return prefac*(1-(2/x**2) + (1+(2/x**2))*exp(-x**2) )
    
    
# Gives the value of lambda_C for a given Lambda and r_C
def lambc(r_C,Lambda,expt):
    """
    Finds the value of lambda_c as a function of r_C dependent on the decoherence strength

    Parameters
    ----------
    r_C : Float
        value of r_C at which to calculate.
    Lambda : Float
        Lambda Parameter governing stregnth of the decoherence.
    expt : class
        Experimental parameters

    Returns
    -------
    Float
        the value of lambda_C for a given r_c and Lambda.

    """

    # # The actual function for lambda_c, based on equations from [Gasbarri PRA (2021)] and [Kaltenbaek EJP Quant Tech 3 (2016)]
    # k = expt.t1*expt.t2/(expt.tTot*expt.tT)
    # # num = -Lambda *(k*expt.d)**2
    # # C   = 108/sqrt(pi)*(expt.mass/m0)**2*(r_C/expt.Radius)**2
    # # den = C*integrate(lambda a: exp(-a**2)*J1(a*expt.Radius/r_C)**2,*integral_limits)* (f(h*expt.t2/(expt.mass*expt.D),r_C,expt)-1)
    
    # For n = 1
    x = h*expt.t2/(expt.mass*expt.D)
    k = expt.t1*expt.t2/(expt.tTot*expt.tT)
    A = 36/sqrt(pi)*(expt.mass/m0)**2*(r_C/expt.Radius)**2
    G_int = integrate(lambda a: exp(-a**2)*J1(a*expt.Radius/r_C)**2,*integral_limits)
    f_int = integrate(lambda a: exp(-a**2)*J1(a*expt.Radius/r_C)**2*Si(a*x/r_C)/a,*integral_limits)
    
    prefactor = -Lambda *(k*expt.d)**2/(A*G_int)
    
    return prefactor/(r_C*f_int/(x*G_int) -1)
    
    
    
    #return num/den
    
    
