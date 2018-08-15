# Transient_Simuulation Methods 

------> See cta_transient_search from lena-lin (https://github.com/lena-lin/cta_transient_search)

------> Preliminary work and tests here, application in upper project

Structure:
### ## Jupyter notebooks ## 
- Fermi und Swift Daten.ipynb : collection of tools for reading and plotting data from Fermi and Swift 
- Lightcurve_Fit.ipynb &  Notebooks for plotting and fitting satellite measured Ligghtcurves of fast transients 
- Relaistic_Expactations.ipynb : Simulation of ground-based GRB observation with CTA based on Gilmore, 2012 
- Spectra_Fits.ipynb : collection of methods for plotting and fitting the spectrum of fast transients 
- Telescope_Positions.ipynb : Collection and plotting of telescope positions in gamma, radio, optical and neutrinos 
- Plots.ipynb : Collection of most important results 

### ## .py used functions ##
- EBL.py : Extrapolation via Dominguez, 2011 up to z=2 
- Fitting_models.py : Collection of models fitting the spectrum, including: Band_function, Powerlaw, LogParabola,.... 
 		      Each for different energies and with the possibility for integration 
- Gilmore_simulation.py : Collection of certain functions to crosscheck Gilmore's model: Redshift distribution plotting, Calculation of significance with Li&Ma,....
- Lightcurve.py : Methods for reading and plotting satallite lightcurves, fitting functions and rescalings 
- Sensitivity.py : Plotting method for CTA's sensitivity, here for South, 20 degree zenith and 1800 seconds 
- spectra.py and spetra_LAT.py : collections of methods for plotting and extrapolating satellite measured spectral shapes 

### ## Folders ## 
- DATA : Storage of used data files as .txt,.fits, etc. with information on Lightcurves
- Kataloge: Storage of used catalogs like FERMI GRBST with information on fitting parameters 
- EBL : Collection of different EBl models and example for the use of EBL Table 
- Sensitivity: CTA's Sensitivity (IRF locally stored, no open data set ! ) 
- cta_transient_search_plots : Plots depending on results of the [above](https://github.com/lena-lin/cta_transient_search) mentioned repo 

