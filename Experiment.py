# -*- coding: utf-8 -*-
"""
Contains all the derived quantites
"""

import numpy as np
from Talbot.spectrum import epsilon_from_data
import json

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
sqrt = np.sqrt

class Experiment:
    
    def __init__(self, name):
        # Load values from named parameters file
        prm_file = f"Scenarios/{name}.json"
        with open(prm_file,"r") as read_file:
            prms = json.load(read_file)
        
        # Convert Mass from AMU to kg then finds particle volume and radius
        self.mass = prms['PartProp']['Mass']*AMU
        self.density = prms['PartProp']['Density']

        # Finds grating spacing grating laser wavenumber from wavelength
        self.lambdaG = prms['LaserProp']['WL']

        # Free-fall times as fraction of Talbot time
        self.a1 = prms['a1']
        self.a2 = prms['a2']     

        self.k_limits, self.epsilon = epsilon_from_data('Scenarios/'+prms['PartProp']['datafile'])

        self.phi0 = prms['Phi0']
        
        self.freq  = prms['LaserProp']['Freq']
        self.T_COM = prms['T_COM']

        self.T_Int = prms['PartProp']['T_INT']
            
        self.T_Env = prms['T_ENV']
        self.pressure = prms['Pressure']
        self.ep0 = prms['PartProp']['Stat_Perm']
        self.Drift = prms['Drift']

        self.grating_photon_absorption = True
        

    # https://realpython.com/python-property/
    @property
    def d(self):
        """Standing wave grating spacing"""
        return self.lambdaG/2

    @property
    def k(self):
        """Grating laser wavenumber"""
        return 2*pi/self.lambdaG

    @property
    def omega(self):
        """Trap angular frequency"""
        return 2*pi*self.freq

    @property
    def sigmav(self): return sqrt(kB*self.T_COM/self.mass)

    @property
    def sigmax(self): return self.sigmav/self.omega

    @property
    def sigmap(self): return self.sigmav*self.mass

    @property
    def tT(self): return self.mass*self.d**2/h

    @property
    def t1(self): return self.tT * self.a1

    @property
    def t2(self): return self.tT * self.a2

    @property
    def tTot(self): return self.t1 + self.t2

    @property
    def eta(self): return self.a1*self.a2/(self.a1+self.a2)

    @property
    def s(self): return self.eta*self.d

    @property
    def Volume(self): return self.mass/self.density

    @property
    def Radius(self): return ((3*self.Volume)/(4*pi))**(1/3)

    @property
    def D(self):
        """Geometrically magnified grating period"""
        return self.d*self.tTot/self.t1

    @property
    def chi(self):
        """Polarizability at grating laser wavelength"""
        e = self.epsilon(self.k)
        return 3*self.Volume*(e - 1)/(e + 2)                

    @property
    def RefInd(self):
        """Refractive index at grating laser wavelength"""
        e = self.epsilon(self.k)
        return sqrt(e)
    
    @property
    def xaxis(self):
        # Values of the x-axis chosen to be one period of the interferance pattern
        #x_lims = self.D/2
        x_lims = 1e-6

        return np.linspace(-x_lims,x_lims,1001)

    def __str__(self):
        particle_str = f'\
            \n   Mass          : {self.mass/AMU:.2e} AMU = {self.mass:.2e} kg \
            \n   Radius        : {self.Radius/1e-9:.2f} nm \
            \n   Volume        : {self.Volume:.2e} m^3 \
            \n   Ref Index     : {self.RefInd.real:.2f}+{self.RefInd.imag:.2f}i\
            \n   Polarizaility : {self.chi.real:.2f}+{self.chi.imag:.2f}i\
            \n   CoM Temp      : {self.T_COM/1e-3:.2f} mK\
            \n   Intermal Temp : {self.T_Int/1e-3:.2f} mK\
            \n   Static Perm   : {self.ep0:.2f} (units)\
            \n   sigma_x       : {self.sigmax/1e-9:.2f} nm\
            \n   sigma_p       : {self.sigmap:.2e} kgm^-1'
            
        laser_str = f'\
            \n   Wavelength         : {self.lambdaG/1e-9:.3f} nm\
            \n   Mag Grating Period : {self.D/1e-9:.2f} nm\
            \n   Wave Number        : {self.k:.2e} m^1'
            
        other_str = f'\
            \n   Pressure      : {self.pressure:.2e} Pa\
            \n   Phase Param   : {self.phi0/np.pi:.2f} pi\
            \n   Trapping Freq : {self.freq/1e3:.2f} kHz\
            \n   Talbot Time   : {self.tT/1e-3:.2f} ms\
            \n   Free fall t1  : {self.a1:.2f} tT\
            \n   Free fall t2  : {self.a2:.2f} tT'
            
        
        return f'\nParticle Properties\
            \n---------------------------------------{particle_str}\
            \n\nLaser Properties\
            \n---------------------------------------{laser_str}\
            \n\nEnviroemtnal and experimental params\
            \n---------------------------------------{other_str}'
        