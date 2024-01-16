#!/usr/bin/env python3

# =============================================================================
# This code generates an NPY file that contains the details about the scattering 
# terms within the Talbot coefficents
# The output NPY is based on the values of kR and ks where: 
# k = 2pi/lambda
# s = a1*a2/(a1+a2) (something like that)
# R is particle radius
# This code should work for the ranges 0<=ks<=50, and 0<=kR<=5
# =============================================================================

import numpy as np
import Talbot.MieScatter as ms
from numpy import cos, sin, pi

Re = np.real
Im = np.imag

# Spherical integration library
# ============================================================================
import quadpy
scheme = quadpy.u3.get_good_scheme(47)

def antipode(theta_phi):
    theta,phi = theta_phi
    return (pi-theta,(phi+pi)%(2*pi))

def Scatter_funcs(ks,kR,RefInd):
    
    # Vector wrapper for the Mie Scattering function so that we can use it in vectorised quadpy library    
    def S1S2(theta):
        try:
            return np.array([S1S2(t) for t in theta])
        except TypeError:
            return ms.MieS1S2(RefInd,kR,cos(theta))
    
    # Finding the scattering amplitude EQ(A6) (1/k term give length scale needed for correct units)
    def f_para_perp(theta_phi):
        theta,phi = theta_phi
        S1,S2 = S1S2(theta).T
        return S1*sin(phi), S2*cos(phi)

    def abFIntegrand(theta_phi):
        theta,phi = theta_phi
        fp_para,fp_perp = f_para_perp(theta_phi)
        fn_para,fn_perp = f_para_perp(antipode(theta_phi))
        
        aIntegrand_para = Re(fp_para.conj()*fn_para)*(cos(ks*cos(theta))-cos(ks))
        bIntegrand_para = Im(fp_para.conj()*fn_para)* sin(ks*cos(theta))
        FIntegrand_para = abs(fp_para)**2*(cos((1-cos(theta))*ks)-1)

        aIntegrand_perp = Re(fp_perp.conj()*fn_perp)*(cos(ks*cos(theta))-cos(ks))
        bIntegrand_perp = Im(fp_perp.conj()*fn_perp)* sin(ks*cos(theta))
        FIntegrand_perp = abs(fp_perp)**2*(cos((1-cos(theta))*ks)-1)

        return np.array([aIntegrand_para, bIntegrand_para, FIntegrand_para,
                         aIntegrand_perp, bIntegrand_perp, FIntegrand_perp])
    
    # Integrate the a, b, and F values
    # ====================================================================================
    a_para,b_para,F_para,a_perp,b_perp,F_perp = scheme.integrate_spherical(abFIntegrand)
    
    return a_para,b_para,F_para,a_perp,b_perp,F_perp

ks = np.linspace(0,50,64)
kR = np.linspace(0,5,101)[1:]
RefInd = (5.6005483050847475+3.0125076271186444j)
from multiprocessing import Pool
if __name__=="__main__":
    data = list()
    with Pool() as p:
        for i,kRi in enumerate(kR):
            print(i,kRi)
            data.append(p.starmap(Scatter_funcs, zip(ks,[kRi]*len(ks),[RefInd]*len(ks))))

    data = np.array(data)

    np.save("NPYs/scattering", data) 
    import matplotlib.pyplot as plt
    titles = ["a_para", "b_para", "F_para", "a_perp", "b_perp", "F_perp"]
    for n,t in enumerate(titles):
        print(n,t)
        plt.figure()
        plt.title(t)
        plt.contourf(ks,kR,data[:,:,n])
        plt.xlabel("ks")
        plt.ylabel("kR")
        plt.savefig(f"{t}.png")
    plt.show()

    from scipy.interpolate import interp2d
    a_para = interp2d(ks,kR,data[:,:,0],bounds_error=True)
    b_para = interp2d(ks,kR,data[:,:,1],bounds_error=True)
    F_para = interp2d(ks,kR,data[:,:,2],bounds_error=True)
    a_perp = interp2d(ks,kR,data[:,:,3],bounds_error=True)
    b_perp = interp2d(ks,kR,data[:,:,4],bounds_error=True)
    F_perp = interp2d(ks,kR,data[:,:,5],bounds_error=True)
    
