import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.integrate import quad
import math
from Fitting_models import  Plaw_LAT, Plaw_CUT, Bandfunc_LAT,Componized_LAT,LogParabola_LAT     ## Fitting models
from EBL import get_tau, get_absorpt ## read EBL model and get absorption factor Tau
from uncertainties import ufloat
from uncertainties import unumpy as unp
from astropy.io import fits


# DataFrame from Fermi LAT GRB catalog: T95,T05, Fluece, Max_Energy, Fitting Model.....
def make_DF_from_LGRB(Path):
    Datei = fits.open(Path,ignore_missing_end=True)
    Katalog = Datei['FERMILGRB']
    Katdf = pd.DataFrame({'Name': Katalog.data['NAME'],
                          'Duration_LAT': Katalog.data['T95_LAT'], 'Duration_LLE': Katalog.data['T95_LLE'],
                          'Trig_Time': Katalog.data['TRIGGER_TIME'],
                          'Fluence': Katalog.data['FLUENCE'],
                          'Emax': Katalog.data['MAX_ENERGY'],
                         'PhotonMax': Katalog.data['MAX_ENERGY_PHOTON_NUMBER'],
                         'Max_Arrive': Katalog.data['MAX_ENERGY_ARRIVAL_TIME'],
                          'PhotonProb': Katalog.data['MAX_ENERGY_PHOTON_PROB'],
                         'Best_Fit': Katalog.data['BEST_FITTING_MODEL'],
                          'Test_Statistic': Katalog.data['TS_MAX']
                         })
    return Katdf

# DataFrame from results of the  first Fermi LAT GRB Catalog (Paper, 2013 [arXiv:1303.2908] )
GRB_EXT = pd.read_csv('Kataloge/GRBs_Joint_Interval.csv',sep=' ',decimal=',')

## Get sepectral indices from tables in LAT GRB paper .
def get_indices_from_LAT(GRB_name):
    GRB = GRB_EXT[GRB_EXT['Name'].str.contains(GRB_name)]
    index = GRB.iloc[0][0]
    Delta_t = ufloat(GRB['Time'][index],0) ## Time between LAT T05 and GBM T95
    BF = GRB['Best_Fit'][index]    ## Best Fit
    MFluence = ufloat(GRB['Main_Fluence'][index],GRB['Main_Fluence_err'][index]) ;
    AFluence = ufloat(GRB['Add_Fluence'][index],GRB['Add_Fluence_err'][index])    ## Fluence bewtween 10 keV and 10 GeV
    ## Band
    BE0 = ufloat(GRB['Band_E0'][index],GRB['Band_E0_err'][index]) ; BA = ufloat(GRB['Band_alpha'][index],GRB['Band_alpha_err'][index]) ;
    BB = ufloat(GRB['Band_beta'][index],GRB['Band_beta_err'][index])
    ## Compton
    CE0 = ufloat(GRB['Compton_E0'][index],GRB['Compton_E0_err'][index]) ; CA = ufloat(GRB['Compton_alpha'][index],GRB['Compton_alpha_err'][index])
    ## Log
    LB = ufloat(GRB['Log_b'][index],GRB['Log_b_err'][index]); LEP = ufloat(GRB['Log_Ep'][index],GRB['Log_Ep_err'][index])
    ## Additional +
    Plaw = ufloat(GRB['Plaw_alpha'][index],GRB['Plaw_alpha_err'][index]) ;  CUT = ufloat(GRB['Plaw_Cut'][index],GRB['Plaw_Cut_err'][index])
    return Delta_t,BF,MFluence,AFluence,BE0,BA,BB,CE0,CA,LB,LEP,Plaw,CUT

## Plot Differential Flux in log-log to energy with best fitting model or if SED = True, Diff. Flux * E²
from ebltable.tau_from_model import OptDepth
tau =  OptDepth.readmodel(model = 'dominguez')

def plot_Flux_Energy_LAT(GRB_name,loglow,loghigh,SED,EBL,Redshift,plot_col):  ## SED = Boolean, EBL = Boolean, then redshift needed
    Delt,BF,MFluence,AFluence,BE0,BA,BB,CE0,CA,LB,LEP,Plaw,CUT  = get_indices_from_LAT(GRB_name)
    E_lines = np.logspace(loglow,loghigh) ## TeV !!!
    Fitpoints = unp.uarray(np.zeros(len(E_lines)),np.zeros(len(E_lines)))
    Factor = 1
    string = r'$\frac{\mathrm{d}N}{\mathrm{d}E}$/ $\frac{1}{\mathrm{cm}²\,\mathrm{s} \, \mathrm{TeV}}$'
    style = '-'
    EBL_note = ''
    if SED == True:
        Factor = E_lines**2
        string = r'$\frac{\mathrm{d}N}{\mathrm{d}E} \cdot$E² / $\frac{\mathrm{TeV}}{\mathrm{cm}²\,\mathrm{s}}$'
    if EBL == True:
        Tau = np.zeros(len(E_lines))
        Tau = np.exp(-1.0*tau.opt_depth(Redshift,E_lines))
        Factor = Factor*Tau
        style = '--'
        EBL_note = '_EBL'

    if BF == 'Comptonized':
        Int = quad(Componized_LAT,1e-8,1e-2,args=(1,CE0.n,CA.n,True))  # Set normalization k to 1 to find true k
        Integral = ufloat(Int[0],Int[1])
        MFl = MFluence/(Delt*Integral)   ## Calculate Flux = true K as an ufloat
        Fitpoints = Componized_LAT(E_lines,MFl,CE0,CA,False)*Factor
        y = unp.nominal_values(Fitpoints) ; yerr = unp.std_devs(Fitpoints)
        plt.plot(E_lines,y, ls = style,linewidth = 0.8,color=plot_col, label='LAT_%s%s'%(BF,EBL_note))
        #plt.fill_between(E_lines,y-yerr,y+yerr,color='crimson',alpha=0.5)
    if BF == 'Comptonized+Plaw':
        Int = quad(Componized_LAT,1e-8,1e-2,args=(1,CE0.n,CA.n,True))
        Integral = ufloat(Int[0],Int[1])
        MFl = MFluence/(Delt*Integral)         ## Calculate Flux
        Int = quad(Plaw_LAT,1e-8,1e-2,args=(1,Plaw.n,True))
        Integral = ufloat(Int[0],Int[1])
        AFl = AFluence/(Delt*Integral)
        Fitpoints = Componized_LAT(E_lines,MFl,CE0,CA,False)*Factor+Plaw_LAT(E_lines,AFl,Plaw,False)*Factor
        y = unp.nominal_values(Fitpoints) ; yerr = unp.std_devs(Fitpoints)
        plt.plot(E_lines,y, ls = style,linewidth = 0.8,color=plot_col, label='LAT_%s%s'%(BF,EBL_note))
        #plt.fill_between(E_lines,y-yerr,y+yerr,color='crimson',alpha=0.5)
    if BF == 'Comptonized+Plaw*Cut':
        Int = quad(Componized_LAT,1e-8,1e-2,args=(1,CE0.n,CA.n,True)) ; Integral = ufloat(Int[0],Int[1])
        MFl = MFluence/(Delt*Integral)
        Int = quad(Plaw_LAT,1e-8,1e-2,args=(1,Plaw.n,True)) ; Integral = ufloat(Int[0],Int[1])
        AFl = AFluence/(Delt*Integral)
        Fitpoints = Componized_LAT(E_lines,MFl,CE0,CA,False)*Factor+Plaw_CUT(E_lines,AFl,Plaw,CUT)*Factor
        y = unp.nominal_values(Fitpoints) ; yerr = unp.std_devs(Fitpoints)
        plt.plot(E_lines,y, ls = style,linewidth = 0.8,color=plot_col, label='LAT_%s%s'%(BF,EBL_note))
        #plt.fill_between(E_lines,y-yerr,y+yerr,color='crimson',alpha=0.5)
    if BF == 'Band':
        Int = quad(Bandfunc_LAT,1e-8,1e-2,args=(1,BA.n,BB.n,BE0.n,True)) ; Integral = ufloat(Int[0],Int[1])
        MFl = MFluence/(Delt*Integral)
        Fitpoints =  Bandfunc_LAT(E_lines,MFl,BA,BB,BE0,False)*Factor
        y = unp.nominal_values(Fitpoints) ; yerr = unp.std_devs(Fitpoints)
        plt.plot(E_lines,y, ls = style,linewidth = 0.8,color=plot_col, label='LAT_%s%s'%(BF,EBL_note))
        #plt.fill_between(E_lines,y-yerr,y+yerr,color='crimson',alpha=0.5)
    if BF == 'Band+Plaw':
        Int = quad(Bandfunc_LAT,1e-8,1e-2,args=(1,BA.n,BB.n,BE0.n,True)) ; Integral = ufloat(Int[0],Int[1])
        MFl = MFluence/(Delt*Integral)
        Intl = quad(Plaw_LAT,1e-8,1e-2,args=(1,Plaw.n,True)) ; Integral = ufloat(Int[0],Int[1])
        AFl = AFluence/(Delt*Integral)
        Fitpoints = Bandfunc_LAT(E_lines,MFl,BA,BB,BE0,False)*Factor+Plaw_LAT(E_lines,AFl,Plaw,False)*Factor
        y = unp.nominal_values(Fitpoints) ; yerr = unp.std_devs(Fitpoints)
        plt.plot(E_lines,y, ls = style,linewidth = 0.8,color=plot_col, label='LAT_%s%s'%(BF,EBL_note))
        #plt.fill_between(E_lines,abs(y-yerr),abs(y+yerr),color='crimson',alpha=0.5)
    if BF == 'Band+Plaw*Cut':
        Int = quad(Bandfunc_LAT,1e-8,1e-2,args=(1,BA.n,BB.n,BE0.n,True))  ; Integral = ufloat(Int[0],Int[1])
        MFl = MFluence/(Delt*Integral)
        Int = quad(Plaw_LAT,1e-8,1e-2,args=(1,Plaw.n,True)) ; Integral = ufloat(Int[0],Int[1])
        AFl = AFluence/(Delt*Integral)
        Fitpoints =  Bandfunc_LAT(E_lines,MFl,BA,BB,BE0,False)*Factor+Plaw_CUT(E_lines,AFl,Plaw,CUT)*Factor
        y = unp.nominal_values(Fitpoints) ; yerr = unp.std_devs(Fitpoints)
        plt.plot(E_lines,y, ls = style,linewidth = 0.8,color=plot_col, label='LAT_%s%s'%(BF,EBL_note))
        #plt.fill_between(E_lines,y-yerr,y+yerr,color='crimson',alpha=0.5)
    if BF == 'LogParabola':
        Int = quad(LogParabola_LAT,1e-8,1e-2,args=(1,LEP.n,LB.n,True),limit=100000) ; Integral = ufloat(Int[0],Int[1])
        MFl = MFluence/(Delt*Integral)
        Fitpoints =LogParabola_LAT(E_lines,MFl,LEP, LB,False)*Factor
        y = unp.nominal_values(Fitpoints) ; yerr = unp.std_devs(Fitpoints)
        plt.plot(E_lines,y, ls = style,linewidth = 0.8,color=plot_col, label='LAT_%s%s'%(BF,EBL_note))
        #plt.fill_between(E_lines,y-yerr,y+yerr,color='crimson',alpha=0.5)

    plt.xscale('log'),plt.yscale('log'),plt.title(GRB_name)
    plt.legend(), plt.xlabel('E / TeV',fontsize=12), plt.ylabel(string,fontsize=12)
