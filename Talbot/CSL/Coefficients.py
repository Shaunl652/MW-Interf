from scipy.constants import Planck as h

def lnR_CSL_func(N,l,r,expt,CSL="Extended"):
    """
    Natural log of the CSL decoherance

    Parameters
    ----------
    N : Integer
        Maximum order on which to calculate.
    l : Array of Float
        2D grid of lambda_C coordinates.
    r : Array of Float
        2D grid of r_C coordinaes.
    expt : class
        Experimental parameters
    CSL : optional
        Specify the type of CSL: None, "Pointlike", "Extended".  Default is "Extended".

    Returns
    -------
    dict
        Natural log of CSL decoherence for each n.

    """
    # Useful Values
    # =======================================================================================================================
    Ns = [*range(N+1)]                   # List of integers from 0 to N

    if CSL == None:
        return {n: 0 for n in Ns}
    if CSL == "Pointlike":
        from Talbot.CSL.Pointlike import Gamma, f
    elif CSL == "Extended":
        from Talbot.CSL.Extended import Gamma, f
    else:
        raise Exception(f"Unknown CSL type: {CSL}")
    
    # Finds the CSL parameters and returns a dictionary for each n
    # =======================================================================================================================
    Gamma_vals = Gamma(l, r, expt)
    return {n: (Gamma_vals*expt.tTot)*(f(n*h*expt.t2/(expt.mass*expt.D),r,expt)-1) for n in Ns[1:]}


