#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
# Plots the Interferance pattern and the number of times a particle is measured 
# in each 'bin'
# =============================================================================

import numpy as np
from Experiment import Experiment
from sys import argv
from Talbot.Likelihood import Like
from random import choices
from collections import Counter

# Reads in the experimental control parameters
expt = Experiment(argv[1])

# Finds the likelihood with no CSL contribution
lkhd = Like(expt,CSL=None)
norm = np.trapz(lkhd,expt.xaxis) # normalisation constant

# Chooses the observation location for k measurments
Xs = choices(expt.xaxis,weights=lkhd,k=10000)

# Count the number of occourances of each x location
counts = Counter(Xs)
Bar_vals = [counts[x] for x in expt.xaxis]


# Plot the graphs together
import matplotlib.pyplot as plt

fig,ax1 = plt.subplots()

colour = 'tab:red'
bar_width = 1.5e-6/1001#expt.xaxis[1]-expt.xaxis[0]
#ax1.bar(expt.xaxis/1e-6,Bar_vals,width=bar_width/1e-6,color=colour) # Plots bar chart
ax1.hist(np.array(Xs)/1e-6,len(expt.xaxis),color=colour)
ax1.set_ylabel('Number of particles measured',color=colour)
ax1.tick_params(axis='y',labelcolor=colour)
ax1.set_xlabel('$z$ [$\mu$m]')


ax2 = ax1.twinx()
colour = 'tab:blue'
ax2.plot(expt.xaxis/1e-6,lkhd/lkhd.max(),color=colour)# plots normalised probability distribution
ax2.set_ylabel('Probability dist. [arb. units]',color=colour)
ax2.tick_params(axis='y',labelcolor=colour)

fig.show()














