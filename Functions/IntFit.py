# -*- coding: utf-8 -*-
"""
Finds the value of Lambda by integtaring above and below the exclusion line
"""

import numpy as np
from Talbot.CSL.Extended import lambc
import json
import matplotlib.pyplot as plt


with open("Space.json","r") as read_file:
    Spc = json.load(read_file)

lam_axis = np.logspace(Spc['Lambda']['Min'],Spc['Lambda']['Max'],Spc['Lambda']['Step'])
r_C_axis = np.logspace(Spc['r_c_ps']['Min'],Spc['r_c_ps']['Max'],Spc['r_c_ps']['Step'])



def FindLambda(Post, P0,Vars,err=0.01,N=100, Plot = False):
    """
    Finds the optimum value of Lambda based on 95(+/-err)% confidance based on integrating

    Parameters
    ----------
    Post : 2D array of float
        Posterior in theta space.
    P0 : Float
        Initial guess for the parameter Lambda.
    Vars : Class
        Contains the derived quantites of the system.
    err : Float
        The maximum allowed error in the confidance. Default is 0.01 (1%)
    N : Integer
        The maximum number of iterations to perform. Default is 100
    Plot : Bool
        Decides if the function should output plots for each loop (the plots are a little ugly). Default is Flase

    Returns
    -------
    P0 : Float
        Best guess of Lambda.
    1-Int : Float
        Confidance of result.

    """
    Full_Integral = np.trapz(np.trapz(Post,lam_axis,axis=0),r_C_axis) # Integral of the full graph
    Post_Up = np.zeros_like(Post)
    Int = 0    # Initial ratio of the integrals
    # Set up the upper and lower bounds, these will get overwritten later as Lambdas are chosen
    LamMax = None
    LamMin = None
    n=0
    while  n<=N :
        n += 1
        Line = lambc(r_C_axis,P0,Vars) # Finds the lambda_c values for each r_c
        
        # Sets the value of the array to 0 if it's below the Lambda line or = Post if above
        for i,r in enumerate(r_C_axis):
            for j,l in enumerate(lam_axis):
                if l <= Line[i]:
                    Post_Up[j,i] = 0
                else:
                    Post_Up[j,i] = Post[j,i]
                    
        Int_Check = np.trapz(np.trapz(Post_Up,lam_axis,axis=0),r_C_axis) # Finds the integral over theta space for Post_Up
        
        Int = Int_Check/Full_Integral # Finds the ratio of the integrals (if 0.05, 5%, then great, otherwise try a new P0)
        
        if Plot:
            plt.pcolormesh(r_C_axis, lam_axis, Post)
            plt.plot(r_C_axis,Line)
            plt.xlabel('$r_c$ [m]'); plt.ylabel('$\lambda_c$ [Hz]')
            plt.xscale('log'); plt.yscale('log')
            plt.title('$\Lambda = ${:.3e}, Interal Ratio = {:.3f}%'.format(P0,Int*100))
            plt.colorbar()
            plt.show()
            plt.close()
        
        if Int > 0.05 + err:
            LamMin = P0
            if LamMax == None:
                P0 *= 10
            else:
                P0 = (LamMax+LamMin)/2
                
        elif Int < 0.05 - err:
            LamMax = P0
            if LamMin == None:
                P0 /= 10
            else:
                P0 = (LamMax+LamMin)/2
        else:
            return P0, 1-Int
    print('Allowed error is too small')
    return P0, 1-Int
