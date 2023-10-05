import numpy as np
def Coef(N,expt,Talbot="Mie",Therm=True):
    """
    Talbot Coefficients including effect of initial thermal state

    Parameters
    ----------
    N : Integer
        Maximum order on which to calculate.
    expt : class
        Experimental parameters
    Talbot : optional
        Specify the type of Talbot coefficients: "Rayleigh" or "Mie".  Default is "Mie".

    Returns
    -------
    dict
        A dictionary containing the coefficients for each value of n.

    """
    
    Ns = [*range(N+1)]
    from numpy import pi, exp
       
    # Calculates the Talbot Coefficents
    # =======================================================================================================================    
    if Talbot=="Rayleigh":
        from Talbot.Grating.Rayleigh import B_n  
    elif Talbot=="Mie":
        from Talbot.Grating.Mie import B_n
    else:
        raise Exception(f"Unknown Talbot coefficient type: {Talbot}")    
    B_dict = {n: B_n(n,expt) for n in Ns}    
    
    # Width of the initial thermal state has a decoherence-like effect
    # =======================================================================================================================
    Thermal = lambda n: -(2*pi**2 * n**2 * expt.sigmax**2 *expt.t2**2)/(expt.d**2 *(expt.t1+expt.t2)**2)
    if Therm:
        lnR_thm = {n : Thermal(n) for n in Ns}
    else:
        lnR_thm = {n : 0 for n in Ns}

    # Return the coefficients as a dictionary
    # =======================================================================================================================
    T_dict = {n: B_dict[n]*exp(lnR_thm[n]) for n in Ns}
    return T_dict
