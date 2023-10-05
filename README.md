# MW-Interf
Simulations of a nearfield Talbot matterwave interferometer such as the one described in [1]. 

The code presented here produces the results of a matterwave interferometry experiment for spherical particles in the Mie regieme. Based on the nearfield Talbot effect, it uses a retroreflected standing wave laser as a phase grating.
The full theory is described in the paper [2].
The results then undergo Bayesian inference [3] to make predictions about the parameters of CSL and quantify the amount of parameter space excluded by a value of Shannon Information [].

To run a file you should be in the main MW-Interf folder then run the code through Python. Note that even when running code in subfolders, e.g. Checks/, you should still run from the main folder, simply pass the folder with the name of the file.
Many files also take arguments, such as the scenario file, these should be passed to the code on the same line that they are run.
E.g. to see the posteriors for a MAQRO-like experiment with a particle of mass $10^8$u, in the Python console run the code <code>%run Simulation.py MAQRO8</code>.

The scenario files are .json files containing the experimental control parameters. These are stored in the Scenarios/ folder. Two example files are given that can be edited to change the experimental scenario.

## Main Folder Code

1) Experiment.py - This will read the data from the Scenario file given and generate a class object containing all the control parameters and derived quantities, such as grating pitch
2) Info_Data.py - This will find the information gained as a function of the number of measured data points for a given scenario, will optimise the values of $phi_0$ and $t_2$ at the start
3) Info_Mass.py - Finds the information gained as a function of particle mass for a given scenario, will optimise the values of $phi_0$ and $t_2$ for each tested mass
4) Interference_Plotting.py - Plots the final interference pattern for a given scenario
5) Optomise.py - Finds the optimum values for $phi_0$ and $t_2$ using the method described in [2]. Can be used as a function from within other code, or can be used on its own when called with a scenario file to optimise and a new scenario file to save the optimum values to
6) Simulation.py - Runs the full simulation for a given scenario and returns probability distributions over the parameter space for various numbers of data points. Must be called with the scenario name and prior type
7) Space.json - Contains the bounds on the parameter space

## Bayesian Folder
This folder contains a number of functions to perform the Bayesian analysis
1) Info.py - Finds the info in a posterior relative to the prior [4]
2) Prior.py - Generates the prior from the likelihood (where applicable) and the given prior type, possible prior types include:
   1) MDIP
   2) Flat
   3) Experimental
3) Updating.py - performs the Bayesian anaylsis from a prior and set of measured data [3]

## Checks Folder
Contains a number of files that we used to check our simulations, included for completeness
1) Bateman_Fig3.py - Reporduces Fig. 3 from [1]
2) Various Belenchia_Fig files - Reproduce the respective figures from the [5]
3) DataPlot.py - Plots the data found in Info_Data.py
4) Decoherence.py - Plots the various sources of decoherence and their effect on the visibility
5) MassPlot.py - Plots the data found in Info_Mass.py
6) OptGraph.py - Plots the optimum values of $\phi_0$ and $t_2$ for a given scenario
 
## Functions Folder
Contains a handful of functions that are used in various places throughout the code
1) InitFit.py - The code that performs the iterative bisection to find the exclusion line

## Scenarios Folder
Contains the various scenario files to be passed to the code

## Talbot Folder
Contains the functions that define the CSL and Talbot effects
1) CSL - contains the CSL-related function in both Mie and Rayleigh regimes
2) Grating - Contains the Talbot terms related function in both regimes
3) Decoherence.py - Calculates the effect of the decoherence terms
4) Likelihood.py - Pulls everything together to find the likelihood $p(z|\vec{\theta})$
5) MieForce.py - Calculates the force on the particle from Mie theory
6) MieScatter.py - Mie scattering code from [6] with some minor corrections
7) spectrum.py - Finds the complex refractive index for the specific material and wavelength combination


