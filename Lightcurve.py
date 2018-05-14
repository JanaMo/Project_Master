import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from astropy.io import fits
import math
from uncertainties import ufloat
from uncertainties import unumpy as unp
from scipy.optimize import curve_fit
import scipy
from scipy import stats
tugreen = '#73ad14'

'''
outsourced code for reading, fitting and plotting light curves using satellite data
'''
#############################################################################################################################################
#														 SWIFT 																			 	#
#############################################################################################################################################
def plot_LC_file(Path_to_lc_file,Sum_Bool):
	'''
	plots the Lightcurve in a.u. vs. Time
	Parameter:
	Path_to_lc_file = Path description where the .lc file is stored
	Sum_bool = boolean if the channals should be summed up
	'''
	File = fits.open(Path_to_lc_file, ignore_missing_end=True)
	R = File['RATE']
	T = R.data['TIME']
	Rate = R.data['RATE']
	C = R.data['TOTCOUNTS']

	plt.plot(T,Rate)
	if Sum_Bool == True:
		Sum = np.zeros(len(Rate))
		for i in range(0,len(Rate)):
		    Sum[i] +=  Rate[i].sum()

		plt.plot(T,Sum,color = 'grey',label='Sum')
		plt.legend()
	plt.show()
	return None

def save_SWIFT_txt(Path_to_lc_file, saved_name):
	'''
	Converts a .lc fits-file into a txt with Exposure and Time
	'''
	File = fits.open(Path_to_lc_file, ignore_missing_end=True)
	R = File['RATE']
	T = R.data['TIME']
	Rate = R.data['RATE']
	Sum = np.zeros(len(Rate))
	for i in range(0,len(Rate)):
		Sum[i] +=  Rate[i].sum()
	data = np.array([T,Sum]) ; data = data.T
	text = 'Left column = Time after Trigger in seconds & Right column = # Counts / Exposure = Flux per cm² s'
	np.savetxt('DATA/LC_SWIFT/%s.txt'%(saved_name),data, fmt=['%f','%f'],header=text )
	return None

#############################################################################################################################################
#														 Fermi 																			 	#
#############################################################################################################################################
def plot_LAT_LC(Path_to_pha_file):
	'''
	plots the Lightcurve in a.u. vs. Time
	Parameter:
	Path_to_lc_file = Path description where the .fits file is stored
	'''
	Datei = fits.open(Path_to_pha_file,ignore_missing_end=True)
	Spec = Datei['SPECTRUM']
	C = Spec.data['COUNTS']
	t = Spec.data['TIME'] ; t = t-t[0]
	E = Spec.data['EXPOSURE']

	Counts = np.zeros(len(C))
	for i in range(0, len(C[0])):
	    Counts += C[:,i]
	Counts = Counts/E
	plt.plot(t,Counts)
	plt.ylabel(r'$\frac{\mathrm{Counts}}{\mathrm{Exposure}}$ / $\frac{1}{cm²s}$')
	plt.xlabel('t / s after trigger')

def save_LAT_txt(Path_to_pha_file, saved_name):
	'''
	Converts a .lc fits-file into a txt with Exposure and Time
	'''
	Datei = fits.open(Path_to_pha_file,ignore_missing_end=True)
	Spec = Datei['SPECTRUM']
	C = Spec.data['COUNTS']
	t = Spec.data['TIME'] ; t = t-t[0]
	E = Spec.data['EXPOSURE']

	Counts = np.zeros(len(C))
	for i in range(0, len(C[0])):
	    Counts += C[:,i]
	Counts = Counts/E
	data = np.array([t[700:1300],Counts[700:1300]]) ; data = data.T
	text = 'Left column = Time after Trigger in seconds & Right column = # Counts / Exposure = Flux per cm² s'
	np.savetxt('%s.txt'%(saved_name),data, fmt=['%f','%f'],header=text )


'''
Different fitting model for the lightcurve's shape
'''
def Gauss(x, a, x0, sigma,b):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))+b

def exp(t,Fp,tp,b,alpha): ## simple Powerlaw + Shift
    return Fp*(t/tp+b)**(-alpha)

def fit_LC_Gaussian_exponential(Path,Offset):
    x,y = np.genfromtxt(Path, unpack=True, skip_header=1 )
    A0 = y.max() ; index = np.argmax(y)
    mu0 = x[index] ## Mitte / Peak, ab hier -1 für exp. Fit
    plt.plot(mu0,y.max(),'*',color='crimson',  ms=10)
    b1 = np.max(y[0:index-8]) ; b2 = np.min(y[0:index-8])
    b0 = np.mean([b1,b2])
    plt.plot(x,y,'-',color='grey', label='$Fermi$-Lightcurve')
    params_e, cov_e = curve_fit(exp, x[index-Offset:index+90],y[index-Offset:index+90],p0 = (1,mu0,1,2),maxfev = 5000) ; errors = np.sqrt(np.diag(cov_e))
    print('Fitergebnisse Gauß: ', params_e)
    xlin_e = np.linspace(x[index-2],x[index+200], 1000)
    plt.plot(xlin_e,exp(xlin_e,*params_e), color=tugreen, label='Exponential fit')
    plt.ylabel(r'$\frac{\mathrm{Counts}}{\mathrm{Exposure}}$ / $\frac{1}{cm²s}$') ; plt.xlabel('t / s after trigger')
    plt.legend()
    Number = Path[24:30] ## Speichere nur Namen ohne ganzen Pfad
    plt.title('GRB%s'%(Number))
    plt.savefig('Plots/Lc_fits/Exponential_%s.pdf' %(Number))
    plt.show() ; plt.clf()
    observed_values= y[index-Offset:index+90] ; expected_values= exp(x[index-Offset:index+90],*params_e)
    Chi,p = stats.chisquare(observed_values, f_exp=expected_values)
    return (params_e,errors,Chi,p)

def fit_LC_simple_Gaussian(Path):
    x,y = np.genfromtxt(Path, unpack=True, skip_header=1 )
    A0 = y.max() ; index = np.argmax(y)
    mu0 = x[index] ## Mitte / Peak, ab hier -1 für exp. Fit
    b1 = np.max(y[0:index-8]) ; b2 = np.min(y[0:index-8])
    b0 = np.mean([b1,b2]) ; y  = y-b0
    y = y+abs(y.min())+1e-3

    plt.plot(x,y,'-',color='grey', label='$Fermi$-measured Lightcurve')
    params, cov = curve_fit(Gauss, x,y,p0=(A0,mu0,3,0)) ;  errors = np.sqrt(np.diag(cov))
    print('Fitergebnisse Gauß: ', params)
    xlin = np.linspace(x.min(),x.max(), 10000)
    plt.plot(xlin,Gauss(xlin,*params),color='crimson', alpha = 0.8, label='Gaussian fit')

    plt.ylabel(r'$\frac{\mathrm{Counts}}{\mathrm{Exposure}}$ / $\frac{1}{cm²s}$') ; plt.xlabel('t / s after trigger')
    plt.legend() ;
    Number = Path[26:33] ## Speichere nur Namen ohne ganzen Pfad
    plt.title('GRB%s'%(Number)) ; plt.savefig('Plots/Lc_fits/Simple_Gauss_%s.pdf' %(Number))
    observed_values= y ; expected_values= Gauss(x,*params)
    Chi,p = stats.chisquare(observed_values, f_exp=expected_values)
    plt.show() ; plt.clf() ; return params,errors,Chi,p


def fit_LC_small_Gaussian(Path):
    x,y = np.genfromtxt(Path, unpack=True, skip_header=1 ) ;
    A0 = y.max() ; index = np.argmax(y)
    mu0 = x[index] ## Mitte / Peak, ab hier -1 für exp. Fit
    b1 = np.max(y[0:index-8]) ; b2 = np.min(y[0:index-8])
    b0 = np.mean([b1,b2]) ; y  = y-b0
    y = y+abs(y.min())+1e-3

    plt.plot(x,y,'-',color='grey', label='$Fermi$-measured Lightcurve')
    params, cov = curve_fit(Gauss, x,y,p0=(1,mu0,1,0)) ; errors = np.sqrt(np.diag(cov))
    print('Fitergebnisse Gauß: ', params)
    xlin = np.linspace(x.min(),x.max(), 10000)
    plt.plot(xlin,Gauss(xlin,*params),color='crimson', alpha = 0.8, label='Gaussian fit')

    plt.ylabel(r'$\frac{\mathrm{Counts}}{\mathrm{Exposure}}$ / $\frac{1}{cm²s}$') ; plt.xlabel('t / s after trigger')
    plt.legend() ;
    Number = Path[26:32] ## Speichere nur Namen ohne ganzen Pfad
    plt.title('GRB%s'%(Number)) ; plt.savefig('Plots/Lc_fits/Small_Gauss_%s.pdf' %(Number))
    observed_values= y ; expected_values= Gauss(x,*params)
    Chi,p = stats.chisquare(observed_values, f_exp=expected_values)
    plt.show() ; plt.clf() ; return params,errors,Chi,p

def rescale_x(Path):
	'''
	Normalize the time component for the fit
	'''
	x,y  =np.genfromtxt(Path,unpack=True)
	mean = x[len(x)-1]-x[0]
	x  = x-x[0]#-mean/2# Time after trigger
	# 600 seconds in simulation
	data = np.array([x,y]) ; data = data.T
	text = 'Left column = Time after Trigger in seconds & Right column = # Counts / Exposure = Flux per cm² s'
	np.savetxt(Path,data, fmt=['%f','%f'],header=text )
	return None

def rescale_y(Path):
	'''
	Normalize the exposure component for the fit
	'''
	x,y  =np.genfromtxt(Path,unpack=True)
	Max = y.max() ; index = np.argmax(y)
	y = y/Max
	data = np.array([x,y]) ; data = data.T
	text = 'Left column = Time after Trigger in seconds & Right column = # Counts / Exposure = Flux per cm² s'
	np.savetxt(Path,data, fmt=['%f','%f'],header=text )
	return None

def reset_txt():
	'''
	Reset Rescaling withour the need for a new download
	'''
	## .txts 2008
	save_LAT_txt('FERMI/LLE_GRBs/gll_cspec_bn080916009_v10.pha', 'DATA/LC/LAT_080916')
	save_LAT_txt('FERMI/LLE_GRBs/gll_cspec_bn081024891_v04.pha', 'DATA/LC/LAT_081024')
    # 2009
	save_LAT_txt('FERMI/LLE_GRBs/gll_cspec_bn090217206_v01.pha', 'DATA/LC/LAT_090217')
	save_LAT_txt('FERMI/LLE_GRBs/gll_cspec_bn090323002_v01.pha', 'DATA/LC/LAT_090323')
	save_LAT_txt('FERMI/LLE_GRBs/gll_cspec_bn090510016_v01.pha', 'DATA/LC/LAT_090510')
	save_LAT_txt('FERMI/LLE_GRBs/gll_cspec_bn090902462_v01.pha', 'DATA/LC/LAT_090902')
	save_LAT_txt('FERMI/LLE_GRBs/gll_cspec_bn090926181_v01.pha', 'DATA/LC/LAT_090926')
	save_LAT_txt('FERMI/LLE_GRBs/gll_cspec_bn091031500_v01.pha', 'DATA/LC/LAT_091031')
    # 2010
	save_LAT_txt('FERMI/LLE_GRBs/gll_cspec_bn100116897_v01.pha', 'DATA/LC/LAT_100116')
	save_LAT_txt('FERMI/LLE_GRBs/gll_cspec_bn100724029_v04.pha', 'DATA/LC/LAT_100724')
	save_LAT_txt('FERMI/LLE_GRBs/gll_cspec_bn100826957_v01.pha', 'DATA/LC/LAT_100826')
    # 2011
	save_LAT_txt('FERMI/LLE_GRBs/gll_cspec_bn110328520_v02.pha', 'DATA/LC/LAT_110328')
	save_LAT_txt('FERMI/LLE_GRBs/gll_cspec_bn110529034_v01.pha', 'DATA/LC/LAT_110529')
	save_LAT_txt('FERMI/LLE_GRBs/gll_cspec_bn110721200_v02.pha', 'DATA/LC/LAT_110721')
	save_LAT_txt('FERMI/LLE_GRBs/gll_cspec_bn110731465_v01.pha', 'DATA/LC/LAT_110731')
    # 2012
	save_LAT_txt('FERMI/LLE_GRBs/gll_cspec_bn120226871_v01.pha', 'DATA/LC/LAT_120226')
	save_LAT_txt('FERMI/LLE_GRBs/gll_cspec_bn120328268_v00.pha', 'DATA/LC/LAT_120328')
	save_LAT_txt('FERMI/LLE_GRBs/gll_cspec_bn120709883_v00.pha', 'DATA/LC/LAT_120709')
	save_LAT_txt('FERMI/LLE_GRBs/gll_cspec_bn120911268_v02.pha', 'DATA/LC/LAT_120911')
	save_LAT_txt('FERMI/LLE_GRBs/gll_cspec_bn121011469_v01.pha', 'DATA/LC/LAT_121011')
	save_LAT_txt('FERMI/LLE_GRBs/gll_cspec_bn121225417_v01.pha', 'DATA/LC/LAT_121225')
    # 2013
	save_LAT_txt('FERMI/LLE_GRBs/gll_cspec_bn130305486_v00.pha', 'DATA/LC/LAT_130305')
	save_LAT_txt('FERMI/LLE_GRBs/gll_cspec_bn130310840_v02.pha', 'DATA/LC/LAT_130310')
	save_LAT_txt('FERMI/LLE_GRBs/gll_cspec_bn130427324_v02.pha', 'DATA/LC/LAT_130427')
	save_LAT_txt('FERMI/LLE_GRBs/gll_cspec_bn130821674_v01.pha', 'DATA/LC/LAT_130821')
	save_LAT_txt('FERMI/LLE_GRBs/gll_cspec_bn131014215_v02.pha', 'DATA/LC/LAT_131014')
	save_LAT_txt('FERMI/LLE_GRBs/gll_cspec_bn131108862_v02.pha', 'DATA/LC/LAT_131108')
	save_LAT_txt('FERMI/LLE_GRBs/gll_cspec_bn131231198_v04.pha', 'DATA/LC/LAT_131231')
    # 2014
	save_LAT_txt('FERMI/LLE_GRBs/gll_cspec_bn140102887_v04.pha', 'DATA/LC/LAT_140102')
	save_LAT_txt('FERMI/LLE_GRBs/gll_cspec_bn140110263_v04.pha', 'DATA/LC/LAT_140110')
	save_LAT_txt('FERMI/LLE_GRBs/gll_cspec_bn140206275_v02.pha', 'DATA/LC/LAT_140206')
	save_LAT_txt('FERMI/LLE_GRBs/gll_cspec_bn140619475_v00.pha', 'DATA/LC/LAT_140619')
	save_LAT_txt('FERMI/LLE_GRBs/gll_cspec_bn141028455_v03.pha', 'DATA/LC/LAT_141028')
	save_LAT_txt('FERMI/LLE_GRBs/gll_cspec_bn141207800_v00.pha', 'DATA/LC/LAT_141207')
	save_LAT_txt('FERMI/LLE_GRBs/gll_cspec_bn141222298_v02.pha', 'DATA/LC/LAT_141222')
    # 2015
	save_LAT_txt('FERMI/LLE_GRBs/gll_cspec_bn150510139_v01.pha', 'DATA/LC/LAT_150510')
	save_LAT_txt('FERMI/LLE_GRBs/gll_cspec_bn150523396_v01.pha', 'DATA/LC/LAT_150523')
	save_LAT_txt('FERMI/LLE_GRBs/gll_cspec_bn150902733_v00.pha', 'DATA/LC/LAT_150902')
	save_LAT_txt('FERMI/LLE_GRBs/gll_cspec_bn151006413_v01.pha', 'DATA/LC/LAT_151006')
    # 2016
	save_LAT_txt('FERMI/LLE_GRBs/gll_cspec_bn160816730_v01.pha', 'DATA/LC/LAT_160816')
	save_LAT_txt('FERMI/LLE_GRBs/gll_cspec_bn160905471_v01.pha', 'DATA/LC/LAT_160905')
	save_LAT_txt('FERMI/LLE_GRBs/gll_cspec_bn160910722_v00.pha', 'DATA/LC/LAT_160910')


def plot_DSSC_curve(Name,Path_to_Source_file):
	'''
	Plot Lightcurve from QQuick look data in .txt files
	'''
	T,Bin,Flux,Err_Flux = np.genfromtxt(Path_to_Source_file, unpack = True,delimiter =',' ,skip_header=1)
	plt.errorbar(T,Flux,xerr=None, yerr=Err_Flux,fmt=None, ecolor='k' ,color='crimson',ms=2,errorevery=2,label='ASDC-Dastenpunkte')#barsabove=True)
	plt.legend()
	plt.xlabel('Time in MJDs')
	plt.ylabel('Flux / $10^{-7}$ Photons/cm² s')
	plt.savefig('Plots/Lightcurve_Fermi_%s.png'%(Name))


def plot_LC_solar_Flare(Path_to_Source_file,saved_name):
	'''
	Plot Solar Flare measured by Fermi
	'''
	Datei = fits.open(Path_to_Source_file,ignore_missing_end=True)
	Rate = Datei['RATE']
	t  =Rate.data['TIME'] ## MET
	t = t-t[0] ## Seconds after Trigger
	E = Rate.data['EXPOSURE']# ~ 25000-75000, in cm² s
	C = Rate.data['COUNTS'] # ~ 2-200
	plt.plot(t,C, label='Total counts')
	plt.legend()
	plt.xlabel('t / s  (in total 1 day)')
	plt.ylabel('Counts')
	Counts = C # /E.mean() ## oder durch E?
	data = np.array([t,Counts]) ; data = data.T
	text = 'Left column = Time after Trigger in seconds & Right column = # Counts'
	np.savetxt('DATA/LC_SOLAR/%s.txt'%(saved_name),data, fmt=['%f','%f'],header=text )



############################################################################ Fermi LAT Catalog PAPER ######################################################
###########################################################################################################################################################
Light = pd.read_csv('Kataloge/Lightcurve_Fit.csv',sep=' ',decimal=',')

@np.vectorize
def simple_Plaw(t,Fp,tp,alpha):
    return Fp*(t/tp)**(-alpha)
@np.vectorize
def broken_Plaw(t,Fp,tb,a1,a2):
	if t>= tb:
			return Fp*(t/tb)**(-a1)
	if t< tb:
    		return Fp*(t/tb)**(-a2)
def plot_Lightcurve(GRBname):
	'''
	plot Lightcurve following a simple or broken Powerlaw with indices measured by Fermi LAT
	'''
	GRB = Light[Light['Name'].str.contains(GRBname)] ; index = GRB.iloc[0][0]
	Fp = ufloat(GRB['Flux'][index],GRB['Flux_err'][index])
	tp = ufloat(GRB['t_peak'][index], GRB['t_peak_err'][index])
	ap = ufloat(GRB['alpha_peak'][index], GRB['alpha_peak_err'][index])
	a95 = ufloat(GRB['alpha_t95'][index], GRB['alpha_t95_err'][index])
	a1 = ufloat(GRB['alpha_1'][index], GRB['alpha_1_err'][index])
	a2 = ufloat(GRB['alpha_2'][index], GRB['alpha_2_err'][index])
	tb = ufloat(GRB['t_break'][index], GRB['t_break'][index])

	t = np.linspace(1*tp,10*tp) ; t_fit = unp.nominal_values(t)
	Fit = simple_Plaw(t,Fp,tp,ap)
	y = unp.nominal_values(Fit);yerr = unp.std_devs(Fit)
	plt.plot(t_fit,y, label='Decline of fitted lightcurve')
	plt.fill_between(t_fit,y-yerr,y+yerr,alpha=0.5, label='Uncertainties')
	plt.axvline(tp.n,label='t$_{\mathrm{peak}}$',linestyle = '--')
	plt.legend() ; plt.yscale('log')
	plt.title('Simple Power Law for %s'%(GRBname)) ; plt.xlabel('$t$ / s after T$_0$') ; plt.ylabel(r'Photon flux $F$ /  $\frac{\gamma}{s cm²}$') ;
	plt.savefig('Plots/Lightcurve_Simple_powerlaw_%s'%(GRBname)); plt.show()

	if tb != 0:
		t = np.linspace(0.1*tb,10*tb);t_fit = unp.nominal_values(t)
		Fit = broken_Plaw(t,Fp,tb,a1,a2)
		y = unp.nominal_values(Fit) ; yerr = unp.std_devs(Fit)
		plt.plot(t_fit,y, label='Decline of fitted lightcurve')
		plt.fill_between(t_fit,y-yerr,y+yerr,alpha=0.5, label='Uncertainties')
		plt.axvline(tp.n,label='t$_{\mathrm{peak}}$',linestyle = '--', alpha = 0.6) ; plt.axvline(tb.n,label='t$_{\mathrm{break}}$', linestyle = '--',alpha=3)
		plt.legend() ; 	plt.yscale('log')
		plt.title('Simple Power Law for %s'%(GRBname)) ; plt.xlabel('$t$ / s after T$_0$') ; plt.ylabel(r'Photon flux $F$ /  $\frac{\gamma}{s cm²}$')
		plt.savefig('Plots/Lightcurve_Broken_powerlaw_%s'%(GRBname));plt.show()
