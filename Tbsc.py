import numpy as np
import Tbsb as tb
from scipy import stats as st

def Tbsc0(c,p):
    
    # constant for Tukey Biweight S 
    try:
        lc=len(c)
    except:
        lc=1

    res=np.zeros(lc)
    for i in range(lc):

        try:
            alpha=c[i]
        except:
            alpha=c
        talpha = np.sqrt(st.chi2.ppf(1-alpha,p))
        maxit = 1000;
        eps = 10**(-8)
        diff = 10**6
        ctest = talpha
        iter0 = 0
        while ((diff>eps) and iter0<maxit):
            cold = ctest
            ctest = tb.Tbsb0(cold,p)/alpha
            diff = abs(cold-ctest)
            iter0 = iter0+1
    res[i]=ctest

    return res
