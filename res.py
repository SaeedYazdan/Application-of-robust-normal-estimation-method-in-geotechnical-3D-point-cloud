import numpy as np
import scipy as sp

import resdis as rd
import fw
import rho

def res0(z,initialmu,initialsigma,k,conv,kp,c,*args):
    # does "k" IRWLS refining steps from "initial.beta" for initial estimator 1
    #
    # if "initial.scale" is present, it's used, o/w the MAD is used
    # kp and c = tuning constants of the equation
    #
    inpu=locals()
    arug=inpu['args']

    n=len(z)
    p=len(z[0])

    rdis = rd.resdis0(z,initialmu,initialsigma)

    if (len(arug) < 8):
        scale1 = np.median(abs(rdis))/.6745
        initialscale = scale1
    else:
        c= arug[0]
        initialscale=arug[1]
        scale1 = initialscale

    if (conv==1):
        k=50

    #if con=1, maximum number of iterations =50

    mu = initialmu
    sigma = initialsigma
    lowerbound = np.median(abs(rdis))/c
    
    numbit=0
    for i in range(k):
        numbit=numbit+1
        # do one step of the iterations to solve for the scale
        scalesuperold = scale1
        scale1 = np.sqrt( scale1**2 * np.mean( rho.rho0(rdis/scale1,c) ) / kp )
        # now do one step of IRWLS with the "improved scale"
        weights = fw.fw0(rdis/scale1,c)        
        W = np.matmul(weights.reshape(len(weights),1) , np.ones((1,p)))
        zw = z* W/np.mean(weights)
        mu1=np.mean(zw, axis=0)
        res=z-(np.ones((n,1))*mu1)
        
        sigma1=np.matmul((res.T),(W*(res)))
        sigma1=(np.linalg.det(sigma1))**(-1/p)*sigma1  
 
        if (np.linalg.det(sigma1)<1e-7): 
            mu1 = initialmu
            sigma1 = initialsigma
            scale1 = initialscale
            break
        # check for convergence
        
        if np.linalg.norm(mu-mu1) / np.linalg.norm(mu) < 1e-4 :
            break
        rdis = rd.resdis0(z,mu1,sigma1)
        mu = mu1
        sigma = sigma1

    #rdis = resdis.resdis0(z,mu,sigma)
    # get the residuals from the last beta
    result={
        "murw" : mu,
        "sigmarw" : sigma,
        "scalerw" : scale1,
        "numbit" : numbit,
        "initialscale" : initialscale        
        }
    return result
