import numpy as np
import rho

def lossS0(u,s,c):
    return np.mean(rho.rho0(u/s,c))
