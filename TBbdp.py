import numpy as np
from scipy import special as sp
import math

def TBbdp0(bdp, p):

    if isinstance(bdp, float):
        lc=1
    else:
        lc=len(bdp)
    c=np.zeros(lc)

    for i in range(lc):
        
        # c = starting point of the iteration
        c[i]=5
        # step = width of the dichotomic search (it decreases by half at each
        # iteration). Generally it can be smaller. A large value ensures converge
        # when bdp is very small and p is very large.
        step=200

        # Convergence condition is E(\rho) = \rho(c) bdp
        #  where \rho(c) for TBW is c^2/6
        Erho1=10


        while abs(Erho1-1)>math.sqrt(2.2204e-16):
            c2=(c[i]**2)/2
            Erho= (p*sp.gammainc(0.5*(p+2),c2)/2-(p**2+2*p)*sp.gammainc(0.5*(p+4),c2)/(4*c2)+\
                    +(p**3+6*p^2+8*p)*sp.gammainc(0.5*(p+6),c2)/(6*(c[i]**4))+ ((c[i]**2)/6)*(1-sp.gammainc(p/2,c2))  )

            if lc==1:
                Erho1=(Erho/(c[i]**2))*(6/bdp)
            else:
                Erho1=(Erho/(c[i]**2))*(6/bdp[i])
    
            step=step/2
            if Erho1>1:
                c[i]=c[i]+step
            else:
                c[i]=max(c[i]-step,0.1)
            print(c[i])
            # disp([step c Erho1])
        # Remark:
        # chi2cdf(x,v) = gamcdf(x,v/2,2) = gammainc(x ./ 2, v/2);
    return c
