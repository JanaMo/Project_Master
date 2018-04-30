import numpy as np
from uncertainties import unumpy as unp
import math
################################################################################
                                # LAT
################################################################################
# Fitting models defined in LAT GRB paper [arXiv:1303.2908]
@np.vectorize
def Plaw_LAT(E,K,Alpha,Integrate):
    if Integrate ==False:
        return K*(E)**(-Alpha)
    if Integrate ==True:
        return K*(E)**(-Alpha)*E
@np.vectorize
def Plaw_CUT(E,K,Alpha,EC):
    return K*(E)**(-Alpha)*unp.exp(-E/EC)
@np.vectorize
def Bandfunc_LAT(E,A,alpha,beta,E0,Integrate):
    if Integrate == True:
        if E<=(alpha-beta)*E0:
            return A*(E)**(alpha)*unp.exp(-E/E0)*E
        else:
            return A*(E)**(beta)*unp.exp(beta-alpha)*((alpha-beta)*E0)**(alpha-beta)*E
    if Integrate == False:
        if E<=(alpha-beta)*E0:
            return A*(E)**(alpha)*unp.exp(-E/E0)
        else:
            return A*(E)**(beta)*unp.exp(beta-alpha)*((alpha-beta)*E0)**(alpha-beta)
@np.vectorize
def Componized_LAT(E,A,E0,alpha,Integrate):
    if Integrate == True:
        return A*(E)**(-alpha)*unp.exp(-E/E0)*E
    else:
        return A*(E)**(-alpha)*unp.exp(-E/E0)

def LogParabola_LAT(E,Sp,Ep,b,Integrate): #Sp describes the Fluence
    if Integrate == True:
        return Sp/(E*E)*10**(-b*unp.log(E/(Ep*Ep)))*E
    else:
         return Sp/(E*E)*10**(-b*unp.log(E/(Ep*Ep)))
def SBPL_LAT(E,k,Epiv,alpha,beta,E0,delta): ##smoothly broken powerlaw
    E1 = E/Epiv ; E2 = E/E0 ; E3 = Epiv/E0 ; index = (alpha+beta)/2
    cos1 = np.cosh(np.log(E2)/delta) ; cos2 = np.cosh(np.log(E3)/delta)
    return k*E1**index*(cos1/cos2)**(index*delta*np.log(10))


################################################################################
                               # GBM
################################################################################
def Plaw(E,K,E0,Alpha):
    return K*(E/E0)**(Alpha)
@np.vectorize
def Bandfunc(E,A,alpha,beta,Ep):
    E0 = Ep/(2+alpha)
    if E<=(alpha-beta)*E0:
        return A*(E/100)**(alpha)*math.exp(-E/E0)
    else:
        return A*(E/100)**(beta)*math.exp(beta-alpha)*(((alpha-beta)*E0)/100)**(alpha-beta)

@np.vectorize
def Bandfunc_GeV(E,A,alpha,beta,Ep):
    if (alpha+2)*(alpha-beta) < 0:
        E0 = -Ep/(2+alpha)
    else:
        E0 = Ep/(2+alpha)
    if E<=(alpha-beta)*E0:
        return A*(E/100*1e9)**(alpha)*unp.exp(-E/E0)
    else:
        return A*(E/100*1e9)**(beta)*unp.exp(beta-alpha)*((((alpha-beta)*E0)/100*1e9)**(alpha-beta))

@np.vectorize
def Comptonized(E,A,Epiv,Ep,alpha):
    return A*(E/Epiv)**(alpha)*math.exp(-(alpha+2)*E/Ep)

@np.vectorize
def SBPL(E,A,Epiv,b,m,Delta,EB):
    q  =np.log10(E/EB)/Delta
    qpiv = np.log10(Epiv/EB)/Delta
    a = m*Delta*np.log((math.exp(q)+math.exp(-q))/2)
    apiv = m*Delta*np.log((math.exp(qpiv)+math.exp(-qpiv))/2)

    return A*(E/Epiv)**b*10**(a-apiv)
