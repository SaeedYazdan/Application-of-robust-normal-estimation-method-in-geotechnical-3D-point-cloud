import numpy as np
import scipy as sp

def resdis0(z,mu,sigma):
    n = len(z)
    central = z - np.ones((n,1))*mu
    sqdis = np.sum((np.matmul(central , np.linalg.inv(sigma))*central), axis=1)
    return sqdis**(0.5)
