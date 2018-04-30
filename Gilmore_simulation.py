import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits  ## pip install astropy --no-deps
from astropy import units as u
from collections import OrderedDict
import scipy
from scipy.integrate import quad


from spectra import make_DF_from_BATSE
from spectra import get_indices_from_BATSE
from spectra import make_DF_from_GBM
from spectra import get_indices_from_GBM
from Fitting_models import Bandfunc_GeV
from Sensitivity import plot_Sens
from Fitting_models import Bandfunc, Comptonized, Plaw, SBPL

from uncertainties import unumpy
from uncertainties import unumpy as unp
from uncertainties import ufloat

from ebltable.tau_from_model import OptDepth
tau =  OptDepth.readmodel(model = 'dominguez')


def integrate_background(Path_to_fits_file):
    '''
    Integrate background events from IRF.
    fits-file provides background rates in 1/MeV/s/sr, therefore energy units must be adapted!
    Parameter:
    bkg = Backgorund entry of Fits.file from cta for certain zenith angle,observation time and telescope configuration
    '''
    cta_perf_fits = fits.open(Path_to_fits_file)
    bkg = cta_perf_fits['BACKGROUND']

    delta_energy = (bkg.data['ENERG_HI'][0] - bkg.data['ENERG_LO'][0]) * u.TeV
    energy = (bkg.data['ENERG_HI'][0] + bkg.data['ENERG_LO'][0])/2 * u.TeV
    delta_x = (bkg.data['DETX_HI'][0] - bkg.data['DETX_LO'][0]) * u.deg
    delta_y = (bkg.data['DETY_HI'][0] - bkg.data['DETY_LO'][0]) * u.deg
    delta_energy, delta_y, delta_x = np.meshgrid(delta_energy, delta_y, delta_x, indexing='ij')
    bin_volume = delta_energy.to(u.MeV) * (delta_y * delta_x).to(u.sr)
    bg = bkg.data['BGD'][0] * (1/(u.MeV * u.s * u.sr))
    integral = bg * bin_volume
    return integral,energy

def calculaterate(Path_to_fits_file,GRBname,BAT_DF,z):
    '''
    Calculate Gamma Rate via extrapolation and folding of BATSE data with CTA's response.
    Parameter:
    Path to fits file with CTA's IRF
    GRBname = Full name or parts of the full name of BATSE named GRBs in BATSE5B catalog
    BAT_DF = Dataframe with all detected GRBs by BATSE
    z = Redshift
    '''

    '''
    Read Fits.File with instruments response
    '''
    cta_perf_fits = fits.open(Path_to_fits_file)
    data_A_eff = cta_perf_fits['EFFECTIVE AREA']

    a_eff_cta = OrderedDict({"E_TeV": (data_A_eff.data['ENERG_LO'][0] + data_A_eff.data['ENERG_HI'][0])/2,
                                "A_eff": data_A_eff.data['EFFAREA'][0]
                            })
    A_eff = a_eff_cta['A_eff'][0]*100*100 # m² to cm²
    E_A_eff = a_eff_cta['E_TeV']

    data_bg_rate = cta_perf_fits['BACKGROUND']
    Energy_Bins = OrderedDict({"Low_E":data_bg_rate.data['ENERG_LO'][0],
                              "High_E": data_bg_rate.data['ENERG_HI'][0]})
    E_Back = (Energy_Bins['Low_E']+Energy_Bins['High_E'])/2

    A_eff = np.interp(E_Back,E_A_eff,A_eff) ## Different Energy bins in A_eff and BGD ---> interpolation

    Rate = np.zeros(len(Energy_Bins['Low_E']))
    Int = np.zeros(len(Energy_Bins['Low_E']))
    IntF  =np.zeros(len(Energy_Bins['Low_E']))
    '''
    Integrate BATSE Flux in CTA's energy regime
    '''
    A,AE,alpha,alphaE,beta,betaE,Ep,EpE = get_indices_from_BATSE(GRBname,BAT_DF)
    A = A*1e9 # keV to TeV
    Ep = Ep*1e-9 # keV to TeV
    def Band(E):
        E0 = Ep/(2+alpha)
        if E<=(alpha-beta)*E0:
            return A*(E/(100*1e-9))**(alpha)*np.exp(-E/E0)*np.exp(-1. * tau.opt_depth(z,E))
        else:
            return A*(E/(100*1e-9))**(beta)*np.exp(beta-alpha)*(((alpha-beta)*E0)/(100*1e-9))**(alpha-beta)*np.exp(-1. * tau.opt_depth(z,E))

    ToBe = Band(1e-4)
    if beta > -2:
        beta = -2
    Is = Band(1e-4)
    A = ToBe/Is*A
    for i in range(0,len(Energy_Bins['Low_E'])):
        (Int[i],IntF[i]) = quad(Band,Energy_Bins['Low_E'][i], Energy_Bins['High_E'][i])
    '''
    Fold Integrated flux with effective Area
    '''
    Rate = Int*A_eff
    RateF = IntF*A_eff
    Gamma_Rate = unumpy.uarray(Rate,RateF)
    return Gamma_Rate

def liMa(Non,Noff,alpha):
    '''
    Calculate significance of a measurement with N_on, N_off
    Parameter:
    Non = signal and background in On direction
    Noff = background in Off direction
    alpha  = ratio of time spend fpr on measurements and for off measurements
    '''
    return np.sqrt(2)*unp.sqrt(Non*unp.log((1+alpha)/alpha*(Non/(Non+Noff)))+Noff*unp.log((1+alpha)*(Noff/(Non+Noff))))

def calculatesignificance(GammaRate, BackgroundRate,time,alpha):
    '''
    Calculate significance with LiMa formula
    Parameter:
    GammaRate and BackgroundRate in 1/s
    Time = Observationtime or duration in s
    alpha = Ratio of On source time to off source time t_on/t_off
    '''

    OnSum = ufloat(0,0)
    OffSum = ufloat(0,0)
    for i in range(0,len(GammaRate)-1):
        NSignal = GammaRate[i]*time
        #Nback= BackgroundRate[i].max()*time*u.s
        Nback = ufloat(BackgroundRate[i].max()*time*u.s,0)
        Noff = Nback*1/alpha
        Non = NSignal + Noff
        OnSum += Non
        OffSum += Noff

    significance = liMa(OnSum,OffSum,alpha)
    return significance


def plot_simulation(GRBname,BAT_DF,z):
    '''
    Plot measured sectrum from BATSE and extrapolation with Bandex model and EBl attenuation
    '''
    A,AE,alpha,alphaE,beta,betaE,Ep,EpE = get_indices_from_BATSE(GRBname,BAT_DF)
    A = A*1e9 # keV to TeV
    Ep = Ep*1e-9 # keV to TeV
    if alpha == -2:
        alpha = -1.99
    xlin = np.logspace(-9,-4.) # Tev
    plt.plot(xlin,Bandfunc_GeV(xlin,A,alpha,beta,Ep)*xlin*xlin, color='k',label='Measured spectrum')
    xlin = np.logspace(-4,4) # Tev
    ToBe = Bandfunc_GeV(10**(-4),A,alpha,beta,Ep)
    '''
    Work in Progress : Understanding the fixed model
    '''
    plt.plot(xlin,Bandfunc_GeV(xlin,A,alpha,beta,Ep)*xlin*xlin+Plaw(xlin,10*A,10e-11,-2)*xlin*xlin, '--',color='#73ac14',label='Extrapolation: Fixed model')
    plt.plot(xlin,Bandfunc_GeV(xlin,A,alpha,beta,Ep)*xlin*xlin*np.exp(-1. * tau.opt_depth(z,xlin))+Plaw(xlin,10*A,10e-11,-2)*xlin*xlin*np.exp(-1. * tau.opt_depth(z,xlin)), '-',color='#73ac14'
             ,label='Fixed model & EBL')
    if beta > -2:
        beta = -2
    Is = Bandfunc_GeV(10**(-4),A,alpha,beta,Ep)
    A = (Is/ToBe)**(-1)*A
    plt.plot(xlin,Bandfunc_GeV(xlin,A,alpha,beta,Ep)*xlin*xlin, '--',color='indigo',label='Extrapolation: Bandex model')
    plt.plot(xlin,Bandfunc_GeV(xlin,A,alpha,beta,Ep)*xlin*xlin*np.exp(-1. * tau.opt_depth(z,xlin)), '-',color='indigo',label='Bandex model & EBL')
    plt.xscale('log') ; plt.yscale('log')
    plt.xlabel('E / TeV')
    plt.ylabel(r'$\frac{\mathrm{d}N}{\mathrm{d}E} \cdot$E² / $\frac{\mathrm{keV}}{\mathrm{cm}²\,\mathrm{s}}$')
    plot_Sens('Tev','binwise')
    plt.ylim(1e-15,1e-4)
    plt.legend() ; plt.show() ; plt.clf()




def plot_Flux_Energy(GRB_name,Tabelle,EBL,Redshift,plot_col):
    BF,K_F,Alpha_F, E0_F,A_F,alpha_F,beta_F,Ep_F,A_C_F,Epiv_F,Ep_C_F,alpha_C_F,A_S_F,Epiv_S_F ,lam1,lam2,EB_F,BS_F = get_indices_from_GBM(GRB_name,Tabelle)
    E_lines = np.logspace(-8,4) ## TeV
    style = '-'
    EBL_note = ''
    Factor = E_lines**2
    string = r'$\frac{\mathrm{d}N}{\mathrm{d}E} \cdot$E² / $\frac{\mathrm{TeV}}{\mathrm{cm}²\,\mathrm{s}}$'
    if EBL == True:
        Tau = np.zeros(len(E_lines))
        Tau =  tau.opt_depth(Redshift,E_lines)
        Factor = Factor*np.exp(-Tau)
        style = '--'
        EBL_note ='_EBL'
    if 'FLNC_PLAW' in BF:
        plt.plot(E_lines,Plaw(E_lines,K_F,E0_F,Alpha_F)*Factor,ls = style ,color=plot_col, label='GBM_PowerLaw%s'%(EBL_note))
    if 'FLNC_BAND' in BF:
        plt.plot(E_lines, Bandfunc(E_lines,A_F,alpha_F,beta_F,Ep_F)*Factor,ls = style ,color=plot_col,label='GBM_Bandfunction%s'%(EBL_note))
    if 'FLNC_COMP' in BF  :
        plt.plot(E_lines, Comptonized(E_lines,A_C_F,Epiv_F,Ep_C_F,alpha_C_F)*Factor,'--',color=plot_col,label='GBM_Comptonized%s'%(EBL_note))
    if 'FLNC_SBPL' in BF: # 'FLNC_SBPL':
        b = (lam1+lam2)/2 ; m=(lam2-lam1)/2
        plt.plot(E_lines, SBPL(E_lines,A_S_F,Epiv_S_F,b,m,BS_F,EB_F)*Factor,ls = style ,color=plot_col,label='GBM_Smoothly broken Plaw%s'%(EBL_note))
    plt.xscale('log'),plt.yscale('log'),plt.title(GRB_name)
    plt.legend(), plt.xlabel('E / GeV',fontsize = 12), plt.ylabel(string, fontsize=12)

def calculaterate_GBM(Path_to_fits_file,GRBname,GBM_DF,z):
    '''
    Calculate Gamma Rate via extrapolation and folding of BATSE data with CTA's response.
    Parameter:
    Path to fits file with CTA's IRF
    GRBname = Full name or parts of the full name of BATSE named GRBs in BATSE5B catalog
    GBM_DF = Dataframe with all detected GRBs by GBM on Fermi
    z = Redshift
    '''

    '''
    Read Fits.File with instruments response
    '''
    print('Start IRF part')
    cta_perf_fits = fits.open(Path_to_fits_file)
    data_A_eff = cta_perf_fits['EFFECTIVE AREA']

    a_eff_cta = OrderedDict({"E_TeV": (data_A_eff.data['ENERG_LO'][0] + data_A_eff.data['ENERG_HI'][0])/2,
                                "A_eff": data_A_eff.data['EFFAREA'][0]
                            })
    A_eff = a_eff_cta['A_eff'][0]*100*100 # m² to cm²
    E_A_eff = a_eff_cta['E_TeV']

    data_bg_rate = cta_perf_fits['BACKGROUND']
    Energy_Bins = OrderedDict({"Low_E":data_bg_rate.data['ENERG_LO'][0],
                              "High_E": data_bg_rate.data['ENERG_HI'][0]})
    E_Back = (Energy_Bins['Low_E']+Energy_Bins['High_E'])/2

    A_eff = np.interp(E_Back,E_A_eff,A_eff) ## Different Energy bins in A_eff and BGD ---> interpolation

    Rate = np.zeros(len(Energy_Bins['Low_E']))
    Int = np.zeros(len(Energy_Bins['Low_E']))
    IntF  =np.zeros(len(Energy_Bins['Low_E']))
    '''
    Integrate GBM Flux in CTA's energy regime
    '''
    print('Ready')
    BF,K_F,Alpha_F, E0_F,A_F,alpha_F,beta_F,Ep_F,A_C_F,Epiv_F,Ep_C_F,alpha_C_F,A_S_F,Epiv_S_F ,lam1,lam2,EB_F,BS_F = get_indices_from_GBM(GRBname,GBM_DF)
    if 'FLNC_PLAW' in BF:
        def Plaww(E):
            return Plaw(E,K_F,E0_F,Alpha_F)*np.exp(-1. * tau.opt_depth(z,E))
        for i in range(0,len(Energy_Bins['Low_E'])):
            (Int[i],IntF[i]) = quad(Plaww,Energy_Bins['Low_E'][i], Energy_Bins['High_E'][i])
    if 'FLNC_BAND' in BF:
        def Band(E):
                return Bandfunc(E,A_F,alpha_F, beta_F, Ep_F)*np.exp(-1. * tau.opt_depth(z,E))
        for i in range(0,len(Energy_Bins['Low_E'])):
            (Int[i],IntF[i]) = quad(Band,Energy_Bins['Low_E'][i], Energy_Bins['High_E'][i])
    if 'FLNC_COMP' in BF  :
        def Comp(E):
                return Comptonized(E,A_C_F,Epiv_F, Ep_C_F, alpha_C_F)*np.exp(-1. * tau.opt_depth(z,E))
        for i in range(0,len(Energy_Bins['Low_E'])):
            (Int[i],IntF[i]) = quad(Comp,Energy_Bins['Low_E'][i], Energy_Bins['High_E'][i])
    if 'FLNC_SBPL' in BF: # 'FLNC_SBPL':
        b = (lam1+lam2)/2 ; m=(lam2-lam1)/2
        def SBPLaw(E):
                return SBPL(E,A_S_F,Epiv_S_F,b,m,BS_F,EB_F)*np.exp(-1. * tau.opt_depth(z,E))
        for i in range(0,len(Energy_Bins['Low_E'])):
            (Int[i],IntF[i]) = quad(SBPLaw,Energy_Bins['Low_E'][i], Energy_Bins['High_E'][i])

    '''
    Fold Integrated flux with effective Area
    '''
    Rate = Int*A_eff
    RateF = IntF*A_eff
    Gamma_Rate = unumpy.uarray(Rate,RateF)
    return Gamma_Rate


'''
WORK in PROGRESS
''' 


def calculaterate_and_Plot_Joint(Path_to_fits_file,GRBname,GBM_DF,z):
    '''
    Calculate Gamma Rate via extrapolation and folding of Joint LAT and GBM  data with CTA's response.
    Parameter:
    Path to fits file with CTA's IRF
    GRBname = Full name or parts of the full name of BATSE named GRBs in BATSE5B catalog
    GBM_DF = Dataframe with all detected GRBs by GBM on Fermi
    z = Redshift
    '''

    '''
    Read Fits.File with instruments response
    '''
    print('Start IRF part')
    cta_perf_fits = fits.open(Path_to_fits_file)
    data_A_eff = cta_perf_fits['EFFECTIVE AREA']

    a_eff_cta = OrderedDict({"E_TeV": (data_A_eff.data['ENERG_LO'][0] + data_A_eff.data['ENERG_HI'][0])/2,
                                "A_eff": data_A_eff.data['EFFAREA'][0]
                            })
    A_eff = a_eff_cta['A_eff'][0]*100*100 # m² to cm²
    E_A_eff = a_eff_cta['E_TeV']

    data_bg_rate = cta_perf_fits['BACKGROUND']
    Energy_Bins = OrderedDict({"Low_E":data_bg_rate.data['ENERG_LO'][0],
                              "High_E": data_bg_rate.data['ENERG_HI'][0]})
    E_Back = (Energy_Bins['Low_E']+Energy_Bins['High_E'])/2

    A_eff = np.interp(E_Back,E_A_eff,A_eff) ## Different Energy bins in A_eff and BGD ---> interpolation

    Rate = np.zeros(len(Energy_Bins['Low_E']))
    Int = np.zeros(len(Energy_Bins['Low_E']))
    IntF  =np.zeros(len(Energy_Bins['Low_E']))
    '''
    Integrate GBM Flux in CTA's energy regime
    '''
    print('Ready')
    BF,K_F,Alpha_F, E0_F,A_F,alpha_F,beta_F,Ep_F,A_C_F,Epiv_F,Ep_C_F,alpha_C_F,A_S_F,Epiv_S_F ,lam1,lam2,EB_F,BS_F = get_indices_from_GBM(GRBname,GBM_DF)
    if 'FLNC_PLAW' in BF:
        def Plaww(E):
            return Plaw(E,K_F,E0_F,Alpha_F)*np.exp(-1. * tau.opt_depth(z,E))
        for i in range(0,len(Energy_Bins['Low_E'])):
            (Int[i],IntF[i]) = quad(Plaww,Energy_Bins['Low_E'][i], Energy_Bins['High_E'][i])
    if 'FLNC_BAND' in BF:
        def Band(E):
                return Bandfunc(E,A_F,alpha_F, beta_F, Ep_F)*np.exp(-1. * tau.opt_depth(z,E))
        for i in range(0,len(Energy_Bins['Low_E'])):
            (Int[i],IntF[i]) = quad(Band,Energy_Bins['Low_E'][i], Energy_Bins['High_E'][i])
    if 'FLNC_COMP' in BF  :
        def Comp(E):
                return Comptonized(E,A_C_F,Epiv_F, Ep_C_F, alpha_C_F)*np.exp(-1. * tau.opt_depth(z,E))
        for i in range(0,len(Energy_Bins['Low_E'])):
            (Int[i],IntF[i]) = quad(Comp,Energy_Bins['Low_E'][i], Energy_Bins['High_E'][i])
    if 'FLNC_SBPL' in BF: # 'FLNC_SBPL':
        b = (lam1+lam2)/2 ; m=(lam2-lam1)/2
        def SBPLaw(E):
                return SBPL(E,A_S_F,Epiv_S_F,b,m,BS_F,EB_F)*np.exp(-1. * tau.opt_depth(z,E))
        for i in range(0,len(Energy_Bins['Low_E'])):
            (Int[i],IntF[i]) = quad(SBPLaw,Energy_Bins['Low_E'][i], Energy_Bins['High_E'][i])

    '''
    Fold Integrated flux with effective Area
    '''
    Rate = Int*A_eff
    RateF = IntF*A_eff
    Gamma_Rate = unumpy.uarray(Rate,RateF)
    return Gamma_Rate
