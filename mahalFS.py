import numpy as np

def mahalFS0(Y,MU,SIGMA):

    Ytilde = Y-MU

    #tmp=(Ytilde/SIGMA)*Ytilde

    sig_1=np.linalg.inv(SIGMA)
    cross=np.dot(Ytilde,sig_1)
    d=np.sum(cross*Ytilde,axis=1)

    return d
    
