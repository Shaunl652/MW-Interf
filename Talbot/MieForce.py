import numpy as np
from numpy.lib.scimath import sqrt
conj = np.conj
Re = np.real
Im = np.imag
pi = np.pi
exp = np.exp

import Talbot.MieScatter as ms

# Finds the z component of the force from Mie Theory EQ(21)
# ============================================================================


# Just below EQ (A13)
zeta = lambda l,k,z: 0.5*((-1)**l*exp(-1j*k*z)+exp(1j*k*z))

def AB_lm(k,z,L,m):
    # EQ A12  and A13


    A = {l: 1j**(l+1)*sqrt(4*pi*(2*l+1))/(2*sqrt(l*(l+1))) * m*zeta(l+1,k,z) for l in range(1,L+1)}
    B = {l: 1j**(l)  *sqrt(4*pi*(2*l+1))/(2*sqrt(l*(l+1)))    *zeta(l,  k,z) for l in range(1,L+1)}

    return A,B



def F_z(z,k,R,e):
    """
    Finds the longitudinal force acting on the particle in the grating beam

    Parameters
    ----------
    z : Float
        Position of the particle (for F_0 should be -lambda/8).
    k : Float
        Grating laser wave number.
    R : Float
        Radius of the particle.
    e : Complex
        Complex refractive index of the particle.

    Returns
    -------
    Float
        The force acting on the particle in the z direction at position z.

    """

    a_tmp, b_tmp  = ms.Mie_ab(e, k*R) # We *think* that a_tmp[0] stores the value of a_1, which is misleading!
    a = {idx+1:-ai for idx,ai in enumerate(a_tmp)} # Now a[1] stores a_1 as expected; there is no a[0] because there is no a_0.
    b = {idx+1:-bi for idx,bi in enumerate(b_tmp)}

    AP,BP = AB_lm(k,z,len(a), 1)
    AN,BN = AB_lm(k,z,len(a),-1)


    sumspos = np.array([l*(l+2)*sqrt((l-1+1)*(l+1+1)/((2*l+3)*(2*l+1))) * \
                        (2*a[l+1]*AP[l+1]*conj(a[l]*AP[l]) + a[l+1]*AP[l+1]*conj(AP[l]) + AP[l+1]*conj(a[l]*AP[l]) +\
                         2*b[l+1]*BP[l+1]*conj(b[l]*BP[l]) +b[l+1]*BP[l+1]*conj(BP[l])+BP[l+1]*conj(b[l]*BP[l]))\
                            + (2*a[l]*AP[l]*conj(b[l]*BP[l]) + a[l]*AP[l]*conj(BP[l]) + AP[l]*conj(b[l]*BP[l]))\
                                for l in range(1,len(a)) ])

    sumsneg = np.array([l*(l+2)*sqrt((l+1+1)*(l-1+1)/((2*l+3)*(2*l+1))) * \
                        (2*a[l+1]*AN[l+1]*conj(a[l]*AN[l]) + a[l+1]*AN[l+1]*conj(AN[l]) + AN[l+1]*conj(a[l]*AN[l]) +\
                         2*b[l+1]*BN[l+1]*conj(b[l]*BN[l]) +b[l+1]*BN[l+1]*conj(BN[l])+BN[l+1]*conj(b[l]*BN[l]))\
                            - (2*a[l]*AN[l]*conj(b[l]*BN[l]) + a[l]*AN[l]*conj(BN[l]) + AN[l]*conj(b[l]*BN[l]))\
                                for l in range(1,len(a)) ])

    return -np.sum(Im(sumspos + sumsneg))
