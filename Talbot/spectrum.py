#!/usr/bin/env python3

def epsilon_from_data(filename):
    from numpy import pi, array, argsort

    lines = open(filename).readlines()

    ks = list()
    nlist = list()
    klist = list()
    for line in lines:
            values = [v.strip() for v in line.strip().split('\t')]
            wavelength = float(values[0])
            wavenumber = 2*pi/wavelength
            ks.append(wavenumber)
            if values[1] != '-':
                    nlist.append((wavenumber, float(values[1])))
            if values[2] != '-':
                    klist.append((wavenumber, float(values[2])))
    nlist = array(nlist)
    klist = array(klist)

    kMin = max([min(nlist[:,0]), min(klist[:,0])])
    kMax = min([max(nlist[:,0]), max(klist[:,0])])

    from scipy.interpolate import interp1d
    o = argsort(nlist[:,0])
    RefractiveIndex = interp1d(nlist[o,0], nlist[o,1])
    o = argsort(klist[:,0])
    AbsorptionCoeff = interp1d(klist[o,0], klist[o,1])
    epsilon = lambda k: (RefractiveIndex(k) + 1j*AbsorptionCoeff(k))**2

    kfiltered = array(ks[ks.index(kMax):ks.index(kMin)])
    o = argsort(kfiltered)
    kfiltered = kfiltered[o]

    return (kMin, kMax), epsilon

if __name__=='__main__':
    from sys import argv
    try:
        (kMin, kMax), epsilon = epsilon_from_data(argv[1])
    except IndexError:
        raise Exception("Specify CSV file from which to extract spectrum.")

    import numpy as np
    import matplotlib.pyplot as plt
    pi = np.pi

    kaxis = np.linspace(kMin, kMax, 1001)
    evals = epsilon(kaxis)

    complex_refractice_index = np.sqrt(evals)

    plt.figure()
    plt.plot(2*pi/kaxis, complex_refractice_index.real)
    plt.plot(2*pi/kaxis, complex_refractice_index.imag)
    plt.yscale('log')
    plt.xscale('log')
    plt.show()
