import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e as elem

import astropy.units as u
tugreen = '#73ae14'


def plot_Sens(scale,style):
    '''
    Plot Sensitivity of CTA's telescopes for zenith angle, observation times and array configurations
    Here used 1800 seconds of observation time, Southern array and 20 deg zenith angle
    Parameter:
    scale = energy scale of the plot, string in ['Kev', 'Mev', 'Gev', 'Tev']
    style = style of datapoints, string in ['curve', 'binwise']
    '''
    Emin,Emax,S = np.genfromtxt('Sensitivity/CTA-Performance-prod3b-v1-South-20deg-30m-DiffSens.txt', unpack=True,skip_header=10)
    S_J = 1e-7*S
    S_ev = S_J/(elem)
    if scale == 'Kev':
        S = S_ev*1e-3
        Emax *=1e9
        Emin *=1e9
    if scale == 'Gev':
        S = S_ev*1e-9
        Emax *= 1e3
        Emin *= 1e3
    if scale == 'Tev':
        S = S_ev*1e-12
    if scale == 'Mev':
        S = S_ev*1e-6
        Emax *= 1e6
        Emin *= 1e6
    if style == 'curve':
        E_center = (Emax+Emin)/2
        plt.plot(E_center,S,'-', color='dimgray',label='CTA Sensitivity - 30 min')
    if style == 'binwise' :
        E_center = (Emax+Emin)/2
        E_width = E_center-Emin
        plt.errorbar(E_center, S, xerr=E_width, yerr=None,fmt="None" ,ecolor = 'dimgray',color='dimgray',label='CTA Sensitivity - 30 min')

    plt.xscale('log')
    plt.yscale('log')
    plt.legend()


'''
Compareable to CTA's sensitivity: Measurements of Crab Nebual
--> Crab parameters in plot_ctools
--> Crb parameters in HEGRA spectrum 2004
'''

def ctools_Crab(E,SED):
    index = 2.48
    C = 5.7e-10 / (u.cm**2 * u.s * u.TeV)
    E_0 = (0.3e6 * u.MeV).to(u.TeV)
    factor = 1
    if SED == True:
        factor = E*E
    return C*(E/E_0)**(-index)*factor

def HEGRA_Crab(E,SED):
    index = 2.62
    C = 2.83e-11 / (u.cm**2 * u.s * u.TeV)
    E_0 = 1 *u.TeV
    factor = 1
    if SED == True:
        factor = E*E
    return C*(E/E_0)**(-index)*factor


def plot_ctools_Crab(logEmin,logEmax,SED):
    e_Gev  =np.logspace(logEmin,logEmax)
    plt.plot(e_Gev,ctools_Crab(e_Gev,SED),color='dimgray',lw=2, label='ctools Crab spectrum')
    plt.plot(e_Gev,ctools_Crab(e_Gev,SED)*0.1, color='dimgray',lw=2, linestyle = '--', label='10 % ctools Crab')
    plt.plot(e_Gev,ctools_Crab(e_Gev,SED)*10, color='dimgray',lw=2, linestyle = ':', label='1000 % ctools Crab')
    plt.xscale('log') ; plt.yscale('log')  ; plt.xlabel('E / TeV')

def plot_HEGRA_Crab(logEmin,logEmax,SED):
    e_Gev  =np.logspace(logEmin,logEmax)
    plt.plot(e_Gev,HEGRA_Crab(e_Gev,SED),color=tugreen,lw=2, label='HEGRA Crab spectrum')
    plt.plot(e_Gev,HEGRA_Crab(e_Gev,SED)*0.1,color=tugreen,lw=2,linestyle='--', label='10 % HEGRA Crab')
    plt.plot(e_Gev,HEGRA_Crab(e_Gev,SED)*10,color=tugreen,lw=2,linestyle=':', label='1000% HEGRA Crab')
    plt.xscale('log') ; plt.yscale('log')  ; plt.xlabel('E / TeV', fontsize=12)
