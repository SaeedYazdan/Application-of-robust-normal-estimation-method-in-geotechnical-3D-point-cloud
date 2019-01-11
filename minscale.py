import TBrho as tb
import numpy as np


def minscale0(u, c, kc, *args):

    inpu=locals()
    arug=inpu['args']

    ## Beginning of code
    if len(arug)<1:
        initialsc = median(abs(u))/.6745
    else:
        initialsc=arug[0]

    if len(arug)<2:
        tol = 1e-7
    else:
        tol=arug[1]

    if len(arug)<3:
        maxiter = 200
    else:
        maxiter=arug[2]


    sc = initialsc
    loop = 0
    err = 1
    while  (( loop < maxiter ) and (err > tol)):
        # scale step: see equation 7.119 of Huber and Ronchetti, p. 176
        # scalenew = scaleold *(1/n)*\sum  \rho(u_i/scaleold) / kc
        scnew = sc*np.sqrt( np.mean(tb.TBrho0(u/sc,c)) / kc)
    
        # Note that when there is convergence 
        # sqrt( mean(TBrho(u/sc,c)) / kc) tends to 1 (from below)
        # disp([loop sc sqrt( mean(TBrho(u/sc,c)) / kc)])
    
        err = abs(scnew/sc - 1)
        sc = scnew
        # disp(sc)
        loop = loop+1

    return sc

