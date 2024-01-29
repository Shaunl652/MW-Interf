# ============================================================================
# Simulates the experiment and finds the posterior
# When running pass to the function the name of the scenario file you want to use
# =============================================================================


from random import choices
import numpy as np
import matplotlib.pyplot as plt
from Talbot.Likelihood import Like
import Bayesian.Prior as pri
from Talbot.CSL.Extended import lambc
from Functions.IntFit import FindLambda
import sys
import json
from Experiment import Experiment
from time import time
from Talbot.CSL.Coefficients import lnR_CSL_func

import scipy.constants as const
AMU = const.atomic_mass


start_time = time()
# Checks that the right number of commands are passed to the code
if len(sys.argv) !=3:
    sys.exit("Must pass scenario .json file and type of prior")

def Info(Post,Prior):
    # Takes in the NORMALISED posterior and prior and gives the information gained
    Inte = Post * np.nan_to_num(np.log2(Post/Prior)) # finds the integrand
    return trapz(trapz(Inte, lam_axis,axis=0),r_C_axis)


with open("Space.json","r") as read_file:
    Spc = json.load(read_file)

lam_axis = np.logspace(Spc['Lambda']['Min'],Spc['Lambda']['Max'],Spc['Lambda']['Step'])
r_C_axis = np.logspace(Spc['r_c_ps']['Min'],Spc['r_c_ps']['Max'],Spc['r_c_ps']['Step'])


# plots the prior and likelihood
# ===========================================================================================================================

expt = Experiment(sys.argv[1])

# Likelihood for w(x|theta)
print("Finding the liklihood")
lkhd = Like(expt)

# Finds the Prior based on given comands
if sys.argv[2] == "Jeff":
    print("Creating Jefferys' Prior")
    J = pri.Jeff(lkhd,expt.xaxis)
    filename = 'Plots/Jeff/'
elif sys.argv[2] == "Experimental":
    print("Creating prior based on full experimental data")
    J = pri.Experimental()
    filename = 'Plots/Experimental/'
    PriorLine = np.zeros_like(r_C_axis)
    for i,r in enumerate(r_C_axis):
        v = pri.Non_Int_Line(r)
        if v >= 10**Spc['Lambda']['Min'] and v <= 10**Spc['Lambda']['Max']:
            PriorLine[i] = v
        else:
            PriorLine[i] = np.nan
elif sys.argv[2] == "Flat":
    print('Making a flat prior')
    J = np.ones((Spc['Lambda']['Step'],Spc['r_c_ps']['Step']))
elif sys.argv[2] == 'MDIP':
    print('Generating Maximal Data information Prior (MDIP)')
    J = pri.MDIP(lkhd,expt.xaxis)
else:
    sys.exit(f'Unrecognised prior type {sys.argv[2]}')

# Normalises Prior
J /= np.trapz(np.trapz(J,lam_axis,axis=0),r_C_axis)

print("Finding the simulation weightings") 
prob = Like(expt,CSL=None) # Just chooses x based on the likelihood where theta=0


# Builds the line to show the minimum allowed values for the CSL parameters
def line(x):
    if x <= 1e-10:
        return 1e-10
    elif x <= 1e-5:
        return 1e-30 * x**-2
    else:
        return 1e-10 * x**2

ExclusionLine = np.zeros_like(r_C_axis)
for i,r in enumerate(r_C_axis):
    val = line(r)
    if val <= np.max(lam_axis) and val >= np.min(lam_axis):
        ExclusionLine[i] = val
    else:
        ExclusionLine[i] = float('nan')



ExclusionLine = [line(r) for r in r_C_axis]


# Runs Simulation
# ===========================================================================================================================

# Simulate random points
Ns = [0,4000,7000,10000]
from scipy.integrate import trapz
from Bayesian.Info import H as infoH

logJ = np.log(J)
lkhd_dict = {x: lkhd[i] for i,x in enumerate(expt.xaxis)}

Posteriors = dict()
Info_dict  = dict() 
for n in Ns:
    print(n)
    # Random x data
    x_data = choices(expt.xaxis,weights=prob,k=n)
    # Finds the posterior from the data x_data
    logLikelihood = sum((np.log(lkhd_dict[x]) for x in x_data))
    # Introduces a reduction term to avoid numerical errors, is canceld out in the normalisation
    kappa = np.max(logLikelihood)
    logLikelihood = logLikelihood - kappa
    # Finds the posterioir
    Post = np.exp(logJ + logLikelihood)
    # Normalises the posterior
    Evid = trapz(trapz(Post, r_C_axis,axis=1),lam_axis)
    Post /= Evid
    
    Info_dict[n] = Info(Post,J)#infoH(x_data,lkhd,J,expt.xaxis)
    Posteriors[n] = Post

MAX = max([P.max() for P in Posteriors.values()])
MIN = min([P.min() for P in Posteriors.values()])
levels=np.linspace(MIN,MAX,11)/MAX

fig, axes = plt.subplots(1, len(Ns), sharey=True, figsize=((16.2,4)), constrained_layout=True)
titles = ['(' + chr(97+n) + f') N={N}; '+'$\\mathcal{H}$'+f' = {Info_dict[N]:.2f}' for n,N in enumerate(Ns)]
#fig.suptitle(f'Posteriors after N data points for a {expt.mass/AMU:.2e} AMU Particle')
for n,ax,title in zip(Ns,axes,titles):
    Post = Posteriors[n]
    m = ax.pcolormesh(r_C_axis,lam_axis,Post/MAX,vmin=0,vmax=1,linewidth=0,rasterized=True)#ax.contourf(r_C_axis, lam_axis, Post/MAX, levels=levels, extend='min')
    # These 'Adler' values are taken from [Adler J Phys A: Math Theor 40 (2007)]
    ax.errorbar([1e-7,1e-6],[1e-8,1e-6],yerr=[[9.9e-9,9.9e-7],[9.9e-7,9.9e-5]],label="Adler Values",fmt='r.',ecolor='r')
    
    # The GRW value is taken from [Ghirardi PRD 34 (1986)]
    ax.scatter(1e-7, 1e-16, label='GRW Value',c='b')

    # Plots the upper bounds found by the simulation on all except the prior
    if n != 0:
        # Finds the decoherence strength
        LParam, Conf = FindLambda(Post,1e13, expt,err=0.001)
        # Plots the upper bounds from the decoherece strength
        UpPlot = np.zeros_like(r_C_axis)
        for i,r in enumerate(r_C_axis):
            val = lambc(r, LParam, expt)
            # THis bit just keeps the line from being drawn above the surface plot
            if val > 10**Spc['Lambda']['Max']:
                UpPlot[i] = np.nan
            else:
                UpPlot[i] = val
        # Now we can plot the line
        ax.plot(r_C_axis,UpPlot,label=r'$\Lambda$ contour', color='tab:red',linestyle='dashed')
    else:
        LParam = 0

    # This exclusion line is taken from [Toros J Phys A: Math Theor 51 (2018)]
    ax.plot(r_C_axis,ExclusionLine,label='Lower Bound',color='k')
    
    # Plots the prior exclusion line if applicable
    #if sys.argv[2] == "Experimental":
    #    plt.plot(r_C_axis,PriorLine,label='Prior Bounds',color='w')
    ax.set_xlabel('$r_c$ [m]')
    if ax is axes[0]:
        ax.set_ylabel('$\lambda_c$ [Hz]')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_title(title)

handles,labels = ax.get_legend_handles_labels()
order = [3,0,1,2]
handles = [handles[o] for o in order]
labels  = [labels[o]  for o in order]
ax.legend(handles, labels, loc='upper right')
cb = plt.colorbar(m, ax=axes, format='%.1f')
cb.set_ticks(levels)
cb.set_label('Arb. Units')
plt.savefig("Plots/posteriors.png")
plt.savefig("Plots/posteriors.pdf")


end_time = (time()-start_time)/60
print(f'Compleated in {end_time:.2} minutes')
plt.show()
