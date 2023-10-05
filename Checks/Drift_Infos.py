#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
# Want to plot the information as a function of mass for various spacecraft drifts
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
sqrt = np.sqrt


Drift_Vals = ['1e-7','1e-7.5','1e-8','1e-9','1e-10']
Line_labels = ['100nm','30nm','10nm', '1nm', '0.1nm']
colours = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
fill_colour = [plt.cm.Blues(0.9), plt.cm.Oranges(0.9), plt.cm.Greens(0.9), plt.cm.Reds(0.9), plt.cm.Purples(0.9)]

fig,ax = plt.subplots()
for i,val in enumerate(Drift_Vals):
    data = np.load(f'NPYs/Drift_Scenarios/{val}/MDIP/Mass/InfoTheta.npy')
    errs = np.load(f'NPYs/Drift_Scenarios/{val}/MDIP/Mass/ErrsTheta.npy')
    xaxs = np.load(f'NPYs/Drift_Scenarios/{val}/MDIP/Mass/MassTheta.npy')
    
    ax.plot(xaxs,data,label=Line_labels[i],color=colours[i])
    ax.fill_between(xaxs,data+sqrt(errs),data-sqrt(errs),alpha=0.3, edgecolor=fill_colour[i], facecolor=fill_colour[i])
    
ax.legend(loc='upper left')
ax.set(xlabel='Mass [u]',xscale='log',ylabel='$ \\langle \\mathcal{H} \\rangle $')
ax.grid(which='both')
fig.show()