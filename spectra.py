import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from astropy.io import fits
import math
from EBL import get_tau, get_absorpt ## read EBL model and get absorption factor Tau
from Fitting_models import Bandfunc_TeV, Comptonized, Plaw, SBPL

# Dataframe of GBM catalog
def make_DF_from_GBM(Path):
    Datei = fits.open(Path,ignore_missing_end=True)
    Katalog = Datei['FERMIGBRST']
    DF = pd.DataFrame({'Name': Katalog.data['NAME'], 'BF': Katalog.data['FLNC_BEST_FITTING_MODEL'],
                           'T90/s': Katalog.data['T90'],
                             'RA': Katalog.data['RA'], 'DEC': Katalog.data['DEC'],
                             'Uncertainty': Katalog.data['ERROR_RADIUS'],
                           'Fluence': Katalog.data['FLUENCE'], # erg/cm²
                            'K_FL': Katalog.data['FLNC_PLAW_AMPL']*1e9,
                            'E0_FL': Katalog.data['FLNC_PLAW_PIVOT']*1e-9, 'Alpha_FL': Katalog.data['FLNC_PLAW_INDEX'],
                            'A_FL': Katalog.data['FLNC_BAND_AMPL']*1e9, 'alpha_FL': Katalog.data['FLNC_BAND_ALPHA'],
                            'beta_FL': Katalog.data['FLNC_BAND_BETA'], 'EP_FL': Katalog.data['FLNC_BAND_EPEAK']*1e-9,
                            'A_COM_FL': Katalog.data['FLNC_COMP_AMPL']*1e9, 'EP_COM_FL': Katalog.data['FLNC_COMP_EPEAK']*1e-9,
                            'Index_FL': Katalog.data['FLNC_COMP_INDEX'], 'EPIV_FL': Katalog.data['FLNC_COMP_PIVOT']*1e-9,
                            'A_SBPL_FL': Katalog.data['FLNC_SBPL_AMPL']*1e9, 'EPIV_SBPL_FL': Katalog.data['FLNC_SBPL_PIVOT']*1e-9,
                            'Index1_FL': Katalog.data['FLNC_SBPL_INDX1'], 'Index2_FL': Katalog.data['FLNC_SBPL_INDX2'],
                            'EBreak_FL': Katalog.data['FLNC_SBPL_BRKEN']*1e-9, 'BreakScale': Katalog.data['FLNC_SBPL_BRKSC']
                         })
    return DF


## Get sepectral indices from GBM cat.
def get_indices_from_GBM(GRB_name,Tabelle):
    GRB = Tabelle[Tabelle.Name.str.contains(GRB_name)]
    BF = GRB.iloc[0]['BF'] ## Best fitting model over duration of the burst
    ## Power Law over duration
    K_F = GRB.iloc[0]['K_FL']; Alpha_F = GRB.iloc[0]['Alpha_FL']; E0_F =GRB.iloc[0]['E0_FL']
    ## Band Function over duration
    A_F = GRB.iloc[0]['A_FL'];alpha_F = GRB.iloc[0]['alpha_FL'];beta_F = GRB.iloc[0]['beta_FL'];Ep_F = GRB.iloc[0]['EP_FL']
    ## Comptonized Model over duration
    A_C_F = GRB.iloc[0]['A_COM_FL'];Epiv_F = GRB.iloc[0]['EPIV_FL'];Ep_C_F = GRB.iloc[0]['EP_COM_FL'];alpha_C_F = GRB.iloc[0]['Index_FL']
    ## Smoothly broken powerlaw
    A_S_F = GRB.iloc[0]['A_SBPL_FL']; Epiv_S_F = GRB.iloc[0]['EPIV_SBPL_FL']; lam1 = GRB.iloc[0]['Index1_FL']; lam2=GRB.iloc[0]['Index2_FL'] ;
    EB_F  = GRB.iloc[0]['EBreak_FL']; BS_F = GRB.iloc[0]['BreakScale'];
    return BF,K_F,Alpha_F, E0_F,A_F,alpha_F,beta_F,Ep_F,A_C_F,Epiv_F,Ep_C_F,alpha_C_F,A_S_F,Epiv_S_F ,lam1,lam2,EB_F,BS_F

## Dataframe of BATSE5B catalog
def make_DF_from_BATSE(Path):
    Datei = fits.open(Path,ignore_missing_end=True)
    BATSE5 = Datei['BAT5BGRBSP']

    BATSE_DF = pd.DataFrame({'Name': BATSE5.data['NAME'],
                         'RA': BATSE5.data['RA'],'DEC': BATSE5.data['DEC'],
                         'Duration':BATSE5.data['T90'],
                         'Total_Time': BATSE5.data['FLNC_INTEGRATION_TIME'],
                         'Band_Fluence':BATSE5.data['FLNC_BAND_FLNC'],
                         'Band_A': BATSE5.data['FLNC_BAND_AMPL'], # Photons / cm² s keV
                         'Band_A_Err': BATSE5.data['FLNC_BAND_AMPL_ERROR'],
                         'Band_Alpha':BATSE5.data['FLNC_BAND_ALPHA'],
                         'Band_Alpha_Err':BATSE5.data['FLNC_BAND_ALPHA_ERROR'],
                         'Band_Beta':BATSE5.data['FLNC_BAND_BETA'],
                         'Band_Beta_Err':BATSE5.data['FLNC_BAND_BETA_ERROR'],
                         'Band_EP': BATSE5.data['FLNC_BAND_EPEAK'],
                         'Band_EP_Err': BATSE5.data['FLNC_BAND_EPEAK_ERROR'],
                         'Band_Chi': BATSE5.data['FLNC_BAND_CHISQ'],
                         })
    return BATSE_DF

## Get spectral indices from BATSE
def get_indices_from_BATSE(GRB_name,Tabelle):
    GRB = Tabelle[Tabelle.Name.str.contains(GRB_name)]
    ## Band Function over duration
    Flu = GRB.iloc[0]['Band_Fluence'] ;  Time = GRB.iloc[0]['Duration']
    A = GRB.iloc[0]['Band_A'];alpha = GRB.iloc[0]['Band_Alpha'];beta = GRB.iloc[0]['Band_Beta'];Ep = GRB.iloc[0]['Band_EP']
    A_E = GRB.iloc[0]['Band_A_Err'];alpha_E = GRB.iloc[0]['Band_Alpha_Err'];beta_E = GRB.iloc[0]['Band_Beta_Err'];Ep_E = GRB.iloc[0]['Band_EP_Err']
    return A,A_E,alpha,alpha_E,beta,beta_E,Ep,Ep_E,Flu,Time 



def plot_Flux_Energy(GRB_name,Tabelle,loglow,loghigh,SED,EBL,Redshift,plot_col):
    BF,K_F,Alpha_F, E0_F,A_F,alpha_F,beta_F,Ep_F,A_C_F,Epiv_F,Ep_C_F,alpha_C_F,A_S_F,Epiv_S_F ,lam1,lam2,EB_F,BS_F = get_indices_from_GBM(GRB_name,Tabelle)
    E_lines = np.logspace(loglow,loghigh) ## TeV
    Factor = 1
    style = '-'
    EBL_note = ''
    string = r'$\frac{\mathrm{d}N}{\mathrm{d}E}$  / $\frac{1}{\mathrm{cm}²\,\mathrm{s}\, \mathrm{TeV}}$'
    if SED == True:
        Factor = E_lines**2
        string = r'$\frac{\mathrm{d}N}{\mathrm{d}E} \cdot$E² / $\frac{\mathrm{TeV}}{\mathrm{cm}²\,\mathrm{s}}$'
    if EBL == True:
        Tau = np.zeros(len(E_lines))
        Tau = get_absorpt(Redshift,E_lines)
        Factor = Factor*Tau
        style = '--'
        EBL_note ='_EBL'
    if 'FLNC_PLAW' in BF:
        plt.plot(E_lines,Plaw(E_lines,K_F,E0_F,Alpha_F)*Factor,ls = style ,color=plot_col, label='GBM_PowerLaw%s'%(EBL_note))
    if 'FLNC_BAND' in BF:
        plt.plot(E_lines, Bandfunc_TeV(E_lines,A_F,alpha_F,beta_F,Ep_F)*Factor,ls = style ,color=plot_col,label='GBM_Bandfunction%s'%(EBL_note))
    if 'FLNC_COMP' in BF :
        plt.plot(E_lines, Comptonized(E_lines,A_C_F,Epiv_F,Ep_C_F,alpha_C_F)*Factor,'--',color=plot_col,label='GBM_Comptonized%s'%(EBL_note))
    if 'FLNC_SBPL' in BF : # 'FLNC_SBPL':
        b = (lam1+lam2)/2 ; m=(lam2-lam1)/2
        plt.plot(E_lines, SBPL(E_lines,A_S_F,Epiv_S_F,b,m,BS_F,EB_F)*Factor,ls = style ,color=plot_col,label='GBM_Smoothly broken Plaw%s'%(EBL_note))
    plt.xscale('log'),plt.yscale('log'),plt.title(GRB_name)
    plt.legend(), plt.xlabel('E / GeV',fontsize = 12), plt.ylabel(string, fontsize=12)
