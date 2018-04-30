import pandas as pd
import numpy as np
import csv
import math

'''
EBL Model from Dominguez, 2011 based on observational results
Absorption Tau saved in .out file, here splitted to a csv. file with additional .txt's for energy and redshift
Tau(certain z, certain E) ist then calculated via interpolaten

-----> New model up to z=4 in 2017 / early 2018, which is used in newest code file
'''
ebl = pd.read_csv('EBL/ebl2.csv',sep=' ',decimal=',', header=None)
ebl_kat = ebl.astype(float) # converts entered values for Tau to floats
arrayZ = np.genfromtxt('EBL/ebl_z.txt', unpack=True,skip_header=1)
arrayE  = np.genfromtxt('EBL/ebl_energy.txt', unpack=True, skip_header=1)*10  # in GeV
@np.vectorize
def get_tau(z,energy):
    if  z>=2 or z<0.01 :
        return 0
    if (energy<=arrayE[0]) or (energy>=arrayE[len(arrayE)-1]): # in keV
        return 0
    else:
        indexZ = np.argmax(arrayZ>z)
        zlow = arrayZ[indexZ-1] # z smaller than entered z
        zhigh = arrayZ[indexZ] # z bigger than entered z
        indexE=np.argmax(arrayE>energy)
        #indexE = 4
        elow = arrayE[indexE-1]# Energy smaller than entered E
        ehigh = arrayE[indexE] # Energy bigger than entered E
        ebl11 = ebl_kat.iloc[indexE-1][indexZ-1]
        ebl12 = ebl_kat.iloc[indexE-1][indexZ]
        ebl21 = ebl_kat.iloc[indexE][indexZ-1]
        ebl22 = ebl_kat.iloc[indexE][indexZ]
        zfrac = (z-zlow)/(zhigh-zlow)
        efrac = (energy-elow)/(ehigh-elow)
        zinterpol1 = ebl11+zfrac*(ebl12-ebl11)
        zinterpol2 = ebl21+zfrac*(ebl22-ebl21)
        Tau = zinterpol1+efrac*(zinterpol2-zinterpol1)
        return Tau
@np.vectorize
def get_absorpt(z,energy):
    return math.exp(-get_tau(z,energy))
