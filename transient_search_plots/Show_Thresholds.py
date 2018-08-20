from evaluation import evaluations
from transient_alert import make_table
import numpy as np
import matplotlib.pyplot as plt

cmap = plt.cm.get_cmap('viridis')
viridis10 = cmap(0.1)
viridis9 = cmap(0.2)
viridis8 = cmap(0.28)
viridis7 = cmap(0.35)
viridis6 = cmap(0.42)
viridis5 = cmap(0.5)
viridis4 = cmap(0.6)
viridis3= cmap(0.7)
viridis2 = cmap(0.82)
viridis1 = cmap(0.9)
viridis0 = cmap(0.98)

def show_thresholds(N,ranges):
    fp   = np.zeros((ranges-1)*4)
    tp = np.zeros((ranges-1)*4)
    fn = np.zeros((ranges-1)*4)
     ## Alert.hdf5 erstellen
    for j in range(1,(ranges-1)*4+1):
        i = j/4
        make_table('/home/jana/Schreibtisch/Projekt_Master/cta_transient_search/build/n{}_s60_trandom_denoised.hdf5'.format(N),i)
        tp[j-1],fp[j-1],fn[j-1]= evaluations('/home/jana/Schreibtisch/Projekt_Master/cta_transient_search/build/n{}_s60_trandom_trans.hdf5'.format(N),  # Trans
    '/home/jana/Schreibtisch/Projekt_Master/cta_transient_search/build/n{}_s60_trandom_thr{}_alert.hdf5'.format(N,i))


    x  = np.arange(1,ranges,0.25)
    plt.plot(x,fp/N, drawstyle='steps-mid',color='#73ac14', label='False Positives')
    plt.plot(x,fn/N,drawstyle='steps-mid',color=viridis5, label='False Negatives (kein Trigger)')
    plt.plot(x,tp/N,drawstyle='steps-mid',color=viridis10, label= 'True Positives')
    plt.xlabel('Schwellwert Triggerkriterium')
    plt.ylabel('Anteil')
    plt.legend(loc='best')
    #plt.savefig('Results_pdf/n{}_s60_trandom_thresholds.pdf'.format(N))
    plt.savefig('Results_jpeg/n{}_s60_trandom_thresholds.jpeg'.format(N))
    plt.clf() 


#show_thresholds(150,10)
#show_thresholds(244,4)
show_thresholds(104,10)
show_thresholds(100,7)
show_thresholds(200,5) 
show_thresholds(108,10)
