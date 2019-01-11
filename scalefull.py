import numpy as np
import rho

def scalefull0(u, kp, c, *args) :

    inpu=locals()
    arug=inpu['args']
    if len(arug)<1:
        initialsc = np.median(abs(u))/6745
    else:
        initialsc=arug[0]

    # find the scale, full iterations
    max_it = 200
    # magic number alert
    # sc = median(abs(u))/.6745
    sc = initialsc
    i = 0
    eps = 1e-20
    # magic number alert
    err = 1
    while  (( i < max_it ) and (err > eps)):
        sc2 = np.sqrt( sc**2 * np.mean(rho.rho0(u/sc,c)) / kp)
        err =abs(sc2/sc - 1)
        sc = sc2
        i=i+1
        
    return sc
