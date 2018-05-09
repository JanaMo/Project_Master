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
from Fitting_models import Bandfunc_TeV
from Sensitivity import plot_Sens
from Fitting_models import Comptonized, Plaw, SBPL

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
    A,AE,alpha,alphaE,beta,betaE,Ep,EpE,Fluence,t = get_indices_from_BATSE(GRBname,BAT_DF)
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
    Gamma_Rate = unumpy.uarray(Rate,RateF) ## With error resulting from numerical calculations in quad
    return Rate

def liMa(Non,Noff,alpha):
    '''
    Calculate significance of a measurement with N_on, N_off
    Parameter:
    Non = signal and background in On direction
    Noff = background in Off direction
    alpha  = ratio of time spend fpr on measurements and for off measurements
    '''
    return np.sqrt(2)*np.sqrt(Non*np.log((1+alpha)/alpha*(Non/(Non+Noff)))+Noff*np.log((1+alpha)*(Noff/(Non+Noff))))

def calculatesignificance(GammaRate, BackgroundRate,time,alpha):
    '''
    Calculate significance with LiMa formula
    Parameter:
    GammaRate and BackgroundRate in 1/s
    Time = Observationtime or duration in s
    alpha = Ratio of On source time to off source time t_on/t_off
    '''

    OnSum = 0
    OffSum = 0
    for i in range(0,len(GammaRate)-1):
        NSignal = GammaRate[i]*time
        Nback = BackgroundRate[i].max()*time*u.s
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
    A,AE,alpha,alphaE,beta,betaE,Ep,EpE,Fluence,Time = get_indices_from_BATSE(GRBname,BAT_DF)
    Fluence = Fluence/(1.602e-19)*1e-7*1e-12/Time
    print('Fluence measured by BATSE',Fluence)
    A = A*1e9 # keV to TeV
    Ep = Ep*1e-9 # keV to TeV
    if alpha == -2:
        alpha = -1.99
    xlin = np.logspace(-9,-4.) # Tev
    plt.plot(xlin,Bandfunc_TeV(xlin,A,alpha,beta,Ep)*xlin*xlin, color='k',label='Measured spectrum')
    xlin = np.logspace(-4,4) # Tev
    ToBe = Bandfunc_TeV(10**(-4),A,alpha,beta,Ep)
    '''
    fixed model
    '''
    Int_Normalization = quad(Plaw_LAT,1e-8,1e-2,args=(1,-2,True),epsrel=1e-6)[0]  # Set normalization k to 1 to find true k
    K = 0.1*Fluence/(Int_Normalization)
    print('K calaculated for additional Power Law ',K)

    plt.plot(xlin,Bandfunc_TeV(xlin,A,alpha,beta,Ep)*xlin*xlin+Plaw(xlin,K,1e-7,-2)*xlin*xlin, '-.',color='#73ac14',lw=2,label='Extrapolation: Fixed model')
    plt.plot(xlin,Bandfunc_TeV(xlin,A,alpha,beta,Ep)*xlin*xlin*np.exp(-1. * tau.opt_depth(z,xlin))+Plaw(xlin,K,1e-7,-2)*xlin*xlin*np.exp(-1. * tau.opt_depth(z,xlin)), '-',color='#73ac14'
             ,lw=2,label='Fixed model & EBL')

    '''
    Bandex
    '''
    if beta > -2:
        beta = -2
    Is = Bandfunc_TeV(10**(-4),A,alpha,beta,Ep)
    A = (Is/ToBe)**(-1)*A
    plt.plot(xlin,Bandfunc_TeV(xlin,A,alpha,beta,Ep)*xlin*xlin, '--',color='indigo',label='Extrapolation: Bandex model')
    plt.plot(xlin,Bandfunc_TeV(xlin,A,alpha,beta,Ep)*xlin*xlin*np.exp(-1. * tau.opt_depth(z,xlin)), '-',color='indigo',label='Bandex model & EBL')
    plt.xscale('log') ; plt.yscale('log')
    plt.xlabel('E / TeV')
    plt.ylabel(r'$\frac{\mathrm{d}N}{\mathrm{d}E} \cdot$E² / $\frac{\mathrm{TeV}}{\mathrm{cm}²\,\mathrm{s}}$')
    plot_Sens('Tev','binwise')
    plt.ylim(1e-20,1e-1)
    plt.title('%s'%(GRBname))
    plt.legend() ;
    #plt.savefig('Plots/Gilmore_Simulation/BATSE_Extrapolation/%s.jpg'%(GRBname),bbox_inches='tight')
    plt.savefig('Plots/Gilmore_Simulation/BATSE_Extrapolation/%s.pdf'%(GRBname),bbox_inches='tight')
    plt.show() ; plt.clf()




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
        if Alpha_F > -2:
            Alpha_F = -2
            plt.plot(E_lines,Plaw(E_lines,K_F,E0_F,Alpha_F)*Factor,ls = style ,color='#73ac14', label='GBM_PowerLaw_Cut%s'%(EBL_note))

    if 'FLNC_BAND' in BF:
        plt.plot(E_lines, Bandfunc_TeV(E_lines,A_F,alpha_F,beta_F,Ep_F)*Factor,ls = style ,color=plot_col,label='GBM_Bandfunction%s'%(EBL_note))
        if beta_F > -2:
            beta_F = -2
            plt.plot(E_lines, Bandfunc_TeV(E_lines,A_F,alpha_F,beta_F,Ep_F)*Factor,ls = style ,color='#73ac14',label='GBM_Band_Cut%s'%(EBL_note))
    if 'FLNC_COMP' in BF  :
        plt.plot(E_lines, Comptonized(E_lines,A_C_F,Epiv_F,Ep_C_F,alpha_C_F)*Factor,ls=style,color=plot_col,label='GBM_Comptonized%s'%(EBL_note))
    if 'FLNC_SBPL' in BF: # 'FLNC_SBPL':
        b = (lam1+lam2)/2 ; m=(lam2-lam1)/2
        plt.plot(E_lines, SBPL(E_lines,A_S_F,Epiv_S_F,b,m,BS_F,EB_F)*Factor,ls = style ,color=plot_col,label='GBM_Smoothly broken Plaw%s'%(EBL_note))
    plt.xscale('log'),plt.yscale('log'),plt.title(GRB_name)
    plt.ylim(1e-15,1e-3)
    plt.legend(), plt.xlabel('E / TeV',fontsize = 12), plt.ylabel(string, fontsize=12)
    #plt.savefig('Plots/Gilmore_Simulation/GBM_Extrapolation/%s.jpg'%(GRB_name),bbox_inches='tight')
    plt.savefig('Plots/Gilmore_Simulation/GBM_Extrapolation/%s_New.pdf'%(GRB_name),bbox_inches='tight')

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
                return Bandfunc_TeV(E,A_F,alpha_F, beta_F, Ep_F)*np.exp(-1. * tau.opt_depth(z,E))
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
    return Rate


from spectra_LAT import get_indices_from_LAT
from Fitting_models import Componized_LAT,Plaw_LAT,Plaw_CUT,Bandfunc_LAT,LogParabola_LAT
GRB_EXT = pd.read_csv('Kataloge/GRBs_Joint_Interval.csv',sep=' ',decimal=',')

def calculaterate_and_Plot_Joint(Path_to_fits_file,GRBname,z):
    '''
    Calculate Gamma Rate via extrapolation and folding of Joint LAT and GBM  data with CTA's response.
    Simultaneously plot the scenario of the extrapolated differential Flux *E²
    Parameter:
    Path to fits file with CTA's IRF
    GRBname = Full name or parts of the full name of BATSE named GRBs in BATSE5B catalog
    GBM_DF = Dataframe with all detected GRBs by GBM on Fermi
    z = Redshift
    '''

    '''Read Fits.File with instruments response'''
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
    IntF = np.zeros(len(Energy_Bins['Low_E']))

    ''' Integrate and Plot GBM Flux in CTA's energy regime'''

    Delt,BF,MFluence,AFluence,BE0,BA,BB,CE0,CA,LB,LEP,Plaw,CUT  = get_indices_from_LAT(GRBname) ## GeV ---> TeV
    plot_col = 'indigo'

    E_lines = np.logspace(-9,3) ## TeV !!!
    Fitpoints = unp.uarray(np.zeros(len(E_lines)),np.zeros(len(E_lines)))
    style = '-'
    EBL_note = ''
    Factor = E_lines**2
    string = r'$\frac{\mathrm{d}N}{\mathrm{d}E} \cdot$E² / $\frac{\mathrm{TeV}}{\mathrm{cm}²\,\mathrm{s}}$'

    if BF == 'Comptonized':
        Int_Normalization = quad(Componized_LAT,1e-8,1e-2,args=(1,CE0.n,CA.n,True),epsrel=1e-6)  # Set normalization k to 1 to find true k
        Integral = ufloat(Int_Normalization[0],Int_Normalization[1])
        MFl = MFluence/(Delt*Integral)   ## Calculate Flux = true K as an ufloat
        Fitpoints = Componized_LAT(E_lines,MFl,CE0,CA,False)*Factor
        y = unp.nominal_values(Fitpoints) ; yerr = unp.std_devs(Fitpoints)
        plt.plot(E_lines,y*np.exp(-1. * tau.opt_depth(z,E_lines)), ls = '--',linewidth = 0.8,color=plot_col, label='LAT_%s_EBL'%(BF))
        plt.plot(E_lines,y, ls = '-',linewidth = 0.8,color=plot_col, label='LAT_%s'%(BF))

        def Comp(E):
                return Comptonized_LAT(E,MFl.n,CE0.n,CA.n,False)*np.exp(-1. * tau.opt_depth(z,E))
        for i in range(0,len(Energy_Bins['Low_E'])):
            (Int[i],IntF[i]) = quad(Comp,Energy_Bins['Low_E'][i], Energy_Bins['High_E'][i])

    if BF == 'Comptonized+Plaw':
        Int_Normalization = quad(Componized_LAT,1e-8,1e-2,args=(1,CE0.n,CA.n,True),epsrel=1e-6)
        Integral = ufloat(Int_Normalization[0],Int_Normalization[1])
        MFl = MFluence/(Delt*Integral)         ## Calculate Flux

        Int_Normalization = quad(Plaw_LAT,1e-8,1e-2,args=(1,Plaw.n,True),epsrel=1e-6)
        Integral = ufloat(Int_Normalization[0],Int_Normalization[1])
        AFl = AFluence/(Delt*Integral)
        Fitpoints = Componized_LAT(E_lines,MFl,CE0,CA,False)*Factor+Plaw_LAT(E_lines,AFl,Plaw,False)*Factor
        y = unp.nominal_values(Fitpoints) ; yerr = unp.std_devs(Fitpoints)
        plt.plot(E_lines,y*np.exp(-1. * tau.opt_depth(z,E_lines)), ls = '--',linewidth = 0.8,color=plot_col, label='LAT_%s_EBL'%(BF))
        plt.plot(E_lines,y, ls = '-',linewidth = 0.8,color=plot_col, label='LAT_%s'%(BF))
        @np.vectorize
        def Comp_Pl(E):
            return Componized_LAT(E,MFl.n,CE0.n,CA.n,False)*np.exp(-1. * tau.opt_depth(z,E))+Plaw_LAT(E,AFl.n,Plaw.n,False)*np.exp(-1. * tau.opt_depth(z,E))
        for i in range(0,len(Energy_Bins['Low_E'])):
            (Int[i],IntF[i]) = quad(Comp_Pl,Energy_Bins['Low_E'][i], Energy_Bins['High_E'][i],epsrel=1e-6)

    if BF == 'Comptonized+Plaw*Cut':
        Int_Normalization = quad(Componized_LAT,1e-8,1e-2,args=(1,CE0.n,CA.n,True),epsrel=1e-6) ; Integral = ufloat(Int_Normalization[0],Int_Normalization[1])
        MFl = MFluence/(Delt*Integral)
        Int_Normalization = quad(Plaw_LAT,1e-8,1e-2,args=(1,Plaw.n,True)) ; Integral = ufloat(Int_Normalization[0],Int_Normalization[1])
        AFl = AFluence/(Delt*Integral)
        Fitpoints = Componized_LAT(E_lines,MFl,CE0,CA,False)*Factor+Plaw_CUT(E_lines,AFl,Plaw,CUT)*Factor
        y = unp.nominal_values(Fitpoints) ; yerr = unp.std_devs(Fitpoints)
        plt.plot(E_lines,y*np.exp(-1. * tau.opt_depth(z,E_lines)), ls = '--',linewidth = 0.8,color=plot_col, label='LAT_%s_EBL'%(BF))
        plt.plot(E_lines,y, ls = '-',linewidth = 0.8,color=plot_col, label='LAT_%s'%(BF))

        def Comp_PlC(E):
                return Componized_LAT(E,MFl.n,CE0.n,CA.n,False)*np.exp(-1. * tau.opt_depth(z,E))+Plaw_CUT(E,AFl.n,Plaw.n,CUT.n)*np.exp(-1. * tau.opt_depth(z,E))
        for i in range(0,len(Energy_Bins['Low_E'])):
            (Int[i],IntF[i]) = quad(Comp_PlC,Energy_Bins['Low_E'][i], Energy_Bins['High_E'][i])


    if BF == 'Band':
        Int_Normalization = quad(Bandfunc_LAT,1e-8,1e-2,args=(1,BA.n,BB.n,BE0.n,True),epsrel=1e-6) ; Integral = ufloat(Int_Normalization[0],Int_Normalization[1])
        MFl = MFluence/(Delt*Integral)
        Fitpoints =  Bandfunc_LAT(E_lines,MFl,BA,BB,BE0,False)*Factor
        y = unp.nominal_values(Fitpoints) ; yerr = unp.std_devs(Fitpoints)
        plt.plot(E_lines,y*np.exp(-1. * tau.opt_depth(z,E_lines)), ls = '--',linewidth = 0.8,color=plot_col, label='LAT_%s_EBL'%(BF))
        plt.plot(E_lines,y, ls = '-',linewidth = 0.8,color=plot_col, label='LAT_%s'%(BF))
        def Band(E):
                return Bandfunc_LAT(E,MFl.n,BA.n,BB.n,BE0.n,False)*np.exp(-1. * tau.opt_depth(z,E))
        for i in range(0,len(Energy_Bins['Low_E'])):
            (Int[i],IntF[i]) = quad(Band,Energy_Bins['Low_E'][i], Energy_Bins['High_E'][i])

    if BF == 'Band+Plaw':
        Int_Normalization = quad(Bandfunc_LAT,1e-8,1e-2,args=(1,BA.n,BB.n,BE0.n,True),epsrel=1e-6) ; Integral = ufloat(Int_Normalization[0],Int_Normalization[1])
        MFl = MFluence/(Delt*Integral)
        Int_Normalization = quad(Plaw_LAT,1e-8,1e-2,args=(1,Plaw.n,True)) ; Integral = ufloat(Int_Normalization[0],Int_Normalization[1])
        AFl = AFluence/(Delt*Integral)
        Fitpoints = Bandfunc_LAT(E_lines,MFl.n,BA.n,BB.n,BE0.n,False)*Factor+Plaw_LAT(E_lines,AFl.n,Plaw.n,False)*Factor
        y = unp.nominal_values(Fitpoints) ; yerr = unp.std_devs(Fitpoints)
        plt.plot(E_lines,y*np.exp(-1. * tau.opt_depth(z,E_lines)), ls = '--',linewidth = 0.8,color=plot_col, label='LAT_%s_EBL'%(BF))
        plt.plot(E_lines,y, ls = '-',linewidth = 0.8,color=plot_col, label='LAT_%s'%(BF))

        def Band_Pl(E):
                return Bandfunc_LAT(E,MFl.n,BA.n,BB.n,BE0.n,False)*np.exp(-1. * tau.opt_depth(z,E))+Plaw_LAT(E,AFl.n,Plaw.n,False)*np.exp(-1. * tau.opt_depth(z,E))
        for i in range(0,len(Energy_Bins['Low_E'])):
            (Int[i],IntF[i]) = quad(Band_Pl,Energy_Bins['Low_E'][i], Energy_Bins['High_E'][i])

    if BF == 'Band+Plaw*Cut':
        Int_Normalization = quad(Bandfunc_LAT,1e-8,1e-2,args=(1,BA.n,BB.n,BE0.n,True),epsrel=1e-6)  ; Integral = ufloat(Int_Normalization[0],Int_Normalization[1])
        MFl = MFluence/(Delt*Integral)
        Int_Normalization = quad(Plaw_LAT,1e-8,1e-2,args=(1,Plaw.n,True)) ; Integral = ufloat(Int_Normalization[0],Int_Normalization[1])
        AFl = AFluence/(Delt*Integral)
        Fitpoints =  Bandfunc_LAT(E_lines,MFl,BA,BB,BE0,False)*Factor+Plaw_CUT(E_lines,AFl,Plaw,CUT)*Factor
        y = unp.nominal_values(Fitpoints) ; yerr = unp.std_devs(Fitpoints)
        plt.plot(E_lines,y*np.exp(-1. * tau.opt_depth(z,E_lines)), ls = '--',linewidth = 0.8,color=plot_col, label='LAT_%s_EBL'%(BF))
        plt.plot(E_lines,y, ls = '-',linewidth = 0.8,color=plot_col, label='LAT_%s'%(BF))

        def Band_PlC(E):
                return Bandfunc_LAT(E,MFl.n,BA.n,BB.n,BE0.n,False)*np.exp(-1. * tau.opt_depth(z,E))+Plaw_CUT(E,AFl.n,Plaw.n,CUT.n)*np.exp(-1. * tau.opt_depth(z,E))
        for i in range(0,len(Energy_Bins['Low_E'])):
            (Int[i],IntF[i]) = quad(Band_PlC,Energy_Bins['Low_E'][i], Energy_Bins['High_E'][i])

    if BF == 'LogParabola':
        Int_Normalization = quad(LogParabola_LAT,1e-8,1e-2,args=(1,LEP.n,LB.n,True),limit=100000,epsrel=1e-6) ; Integral = ufloat(Int_Normalization[0],Int_Normalization[1])
        MFl = MFluence/(Delt*Integral)
        Fitpoints =LogParabola_LAT(E_lines,MFl,LEP, LB,False)*Factor
        y = unp.nominal_values(Fitpoints) ; yerr = unp.std_devs(Fitpoints)
        plt.plot(E_lines,y*np.exp(-1. * tau.opt_depth(z,E_lines)), ls = '--',linewidth = 0.8,color=plot_col, label='LAT_%s_EBL'%(BF))
        plt.plot(E_lines,y, ls = '-',linewidth = 0.8,color=plot_col, label='LAT_%s'%(BF))

        def Log(E):
                return LogParabola_LAT(E,MFl.n,LEP.n, LB.n,False)*np.exp(-1. * tau.opt_depth(z,E))
        for i in range(0,len(Energy_Bins['Low_E'])):
            (Int[i],IntF[i]) = quad(Log,Energy_Bins['Low_E'][i], Energy_Bins['High_E'][i])

    plt.xscale('log'),plt.yscale('log'),plt.title(GRBname),
    plt.ylim(1e-15,1e-3)
    plt.legend(), plt.xlabel('E / TeV',fontsize=12), plt.ylabel(string,fontsize=12)
    plt.legend() ;
    #plt.savefig('Plots/Gilmore_Simulation/LAT_Extrapolation/%s.jpg'%(GRBname),bbox_inches='tight')
    plt.savefig('Plots/Gilmore_Simulation/LAT_Extrapolation/%s.pdf'%(GRBname),bbox_inches='tight')

    ''' Fold Integrated flux with effective Area '''
    Rate = Int*A_eff
    RateF = IntF*A_eff
    Gamma_Rate = unumpy.uarray(Rate,RateF)
    return Rate # no numerical uncertainy here





################################################################################# Simulations #######################################################################################
def simulate_BATSE_detection(Path_to_fits_file,BATSE_DF, GRBname,z,time):
    print('Start simulation for', GRBname, 'with random redshift = ', z)
    BGD_Rate = integrate_background(Path_to_fits_file)[0] # integrate--- has two outputs
    Gamma_Rate = calculaterate(Path_to_fits_file,GRBname,BATSE_DF,z)
    Sigma = calculatesignificance(Gamma_Rate,BGD_Rate,time,1)
    print(Sigma, 'Sigma reached with a maximum background and EBL absorption')
    print('Duration of simulation in seconds: ',time)
    plot_simulation(GRBname,BATSE_DF,z)
    return Sigma


def simulate_GBM_detection(Path_to_fits_file,GBM_DF, GRBname,z,time):
    print('Start simulation for', GRBname, 'with (random?) redshift = ', z)
    BGD_Rate = integrate_background(Path_to_fits_file)[0] # integrate--- has two outputs
    Gamma_Rate = calculaterate_GBM(Path_to_fits_file,GRBname,GBM_DF,z)
    Sigma = calculatesignificance(Gamma_Rate,BGD_Rate,time,1)
    print(Sigma, 'Sigma reached with a maximum background and EBL absorption')
    print('Duration of simulation in seconds: ',time)
    plot_Sens('Tev', 'curve')
    plot_Flux_Energy(GRBname,GBM_DF,False,z,'indigo')
    plot_Flux_Energy(GRBname,GBM_DF,True,z,'indigo')
    plt.show()
    return Sigma

def simulate_LAT_detection(Path_to_fits_file,GRBname,time,z):
    plot_Sens('Tev', 'binwise')
    Gamma_R = calculaterate_and_Plot_Joint(Path_to_fits_file,GRBname,z)
    plt.ylim(1e-15,1e-3)
    Back_R = integrate_background(Path_to_fits_file)[0]
    Sigma = calculatesignificance(Gamma_R,Back_R,time,1)
    return Sigma
