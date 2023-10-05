import numpy as np
from numpy import exp
from numpy.lib.scimath import sqrt
from scipy.special import gamma, sici,spherical_jn
Si = lambda x: sici(x)[0]
sinc = lambda x: np.sinc(x/np.pi)
J1 = lambda x: spherical_jn(1, x)
import json
from Talbot.MieScatter import MieQ

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

def Deco(N,expt,Seperate=False):
    """
    Finds the magnitiude of the conventional decoherence sources

    Parameters
    ----------
    N : Integer
        Maximum order on which to calculate.
    expt : class
        Experimental parameters

    Returns
    -------
    dict
        Contains the value for the decoherence at each n.

    """

    # Functions used in decoherehce
    k_limits, epsilon = expt.k_limits, expt.epsilon

    #chi = lambda Volume,k: 3*Volume*(epsilon(k) - 1)/(epsilon(k) + 2)
    
    def gamma_vals(kvals,T):
        gamma_abs = []
        gamma_sca = []
        gamma_emi = []
        for k in kvals:
            WL = 2*pi/k
            # Find the cross sections for the given particle size, this is valid in Mie regime
            CrossSections = MieQ(expt.RefInd,WL/1e-9,expt.Radius*2/1e-9,asDict=True,asCrossSection=True)
        
            sigma_abs = CrossSections['Cabs']*1e-18
            sigma_sca = CrossSections['Csca']*1e-18
        
            gabs = (k/pi)**2 * sigma_abs/(exp(hbar*c*k/(kB*T))-1) # SupEq6
            gsca = (k/pi)**2 * sigma_sca/(exp(hbar*c*k/(kB*T))-1) # SupEq7
            #gemi = gamma_abs
            gamma_abs.append(gabs)
            gamma_sca.append(gsca)
            gamma_emi.append(gabs)
        
        return gamma_abs,gamma_sca,gamma_emi

    T_Int = expt.T_Int
    T_Env = expt.T_Env

    Ns = [*range(N+1)]   # List of integers from 0 to N

    # Gas Collisions SupEq32
    # =======================================================================================================================
    # Take most of these values from the suplimentary text
    ep0 = expt.ep0                                           # Complex permitvity at omega = 0 (static permitivity)
    pressure = expt.pressure                                 # Pressure in Pa
    m_g = 28 * AMU                                           # Gas Particle Mass in amu converted to kg
    vel = np.sqrt(3*kB*T_Env/m_g)                            # Mean velocity of Gas Particles
    alpha0 = 3*epsilon_0*expt.Volume * ((ep0 - 1)/(ep0 + 2)) # Polarizability
    I_g = 15.6 * 1.60218e-19                                 # eV converted to joules
    I = 5. * 1.60218e-19                                     # eV converted to joules
    alpha_g = 1.74*(1e-10)**3 * 4*pi*epsilon_0
    CoupleConst = 3*alpha0*alpha_g*I_g*I/(32*pi**2*epsilon_0**2*(I+I_g)) # Van Der Waals coupling constant
    Gamma_col = 4*pi*gamma(9/10)/(5*np.sin(pi/5)) * (3*pi*CoupleConst/(2*hbar))**(2/5)* pressure * vel**(3/5)/(kB*T_Env)

    # Computes the uncertainty in measured position due to movement of the satelite
    # =======================================================================================================================
    dsig_dt = expt.Drift/100
    sigmam = expt.sigmax + dsig_dt*(expt.t1+expt.t2) # Uncertanty in measurment position
    Uncert = lambda n: -0.5*(n*(2*pi/expt.D*sigmam)**2)

    # Builds the decoherence dictionary
    # =======================================================================================================================
    lnR_col = dict()
    lnR_abs = dict()
    lnR_sca = dict()
    lnR_emi = dict()
    lnR_mov = dict()
    lnR_thm = dict()

    kaxis = np.linspace(*k_limits, 1001)
    theta = np.linspace(0,1,2001)
    KAXIS,THETA = np.meshgrid(kaxis,theta)
    gamma_abs,gamma_sca,gamma_emi = gamma_vals(kaxis,T_Env)
    for num in Ns:
        if num == 0:
            lnR_col[0] = 0
            lnR_abs[0] = 0
            lnR_sca[0] = 0
            lnR_emi[0] = 0
            lnR_mov[0] = 0
            lnR_thm[0] = 0
            continue

        lnR_col[num] = -Gamma_col*(expt.t1+expt.t2)

        a_n = kaxis*num*h*expt.t2/(expt.D*expt.mass)
        abs_integrand = gamma_abs*(Si(a_n)/a_n - 1)*(expt.t1+expt.t2)
        lnR_abs[num] = c*np.trapz(abs_integrand, kaxis)

        sca_integrand = gamma_sca*(Si(2*a_n)/a_n - sinc(a_n)**2 - 1)*(expt.t1+expt.t2)
        lnR_sca[num] = c*np.trapz(sca_integrand, kaxis)

        # This is valid if the particle does not cool down significantly
        emi_integrand = gamma_emi*(Si(a_n)/a_n - 1)*(expt.t1+expt.t2)
        lnR_emi[num] = c*np.trapz(emi_integrand, kaxis)

        """
        # This is only valid if the particle cools down signification during flight
        emi_part1 = expt.t1*gamma_emi(expt.Volume, KAXIS,T_Int*(expt.t1-expt.t1*THETA))
        emi_part2 = expt.t2*gamma_emi(expt.Volume, KAXIS,T_Int*(expt.t1+expt.t2*THETA))
        emi_integrand = (emi_part1 + emi_part2)*(sinc(a_n*THETA)-1)
        lnR_emi[num] = c*np.trapz(np.trapz(emi_integrand,theta,axis=0),kaxis)
        """

        lnR_mov[num] = Uncert(num)
    if Seperate:
        return {n: lnR_col[n] for n in Ns}, {n: lnR_abs[n] for n in Ns}, {n: lnR_sca[n] for n in Ns}, {n: lnR_emi[n] for n in Ns}, {n: lnR_mov[n] for n in Ns}
    else:
        return {n: lnR_col[n] + lnR_abs[n] + lnR_sca[n] + lnR_emi[n] + lnR_mov[n] for n in Ns}
