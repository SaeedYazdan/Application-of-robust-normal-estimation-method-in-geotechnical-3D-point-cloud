import numpy as np
import pandas as pd
from scipy.linalg import fractional_matrix_power
from scipy.stats import rankdata
from scipy.stats import norm

import resdis as rs
import DetS as ds
import qn
import ogkscatter as og



def scaler(x, scales):
    if scales=='qn':
        return qn.qn0(x)
    else:
        return ds.W_scale(x)
        

def initnew0(z,ind,scales):

    # ind is the number of the initial estimator (1 to 6)

    #i=1 Hyperbolic tangent of standardized data
    #i=2 Spearman correlation matrix
    #i=3 Tukey normal scores
    #i=4 spherical wisnorization
    #i=5 BACON
    #i=6 Raw OGK estimate for scatter

    n=len(z)
    p=len(z[0])

    #initial estimator 1: y=tanh(z)
    if ind==1:
        y1=np.tanh(z)
        k=pd.DataFrame(y1)
        R=k.corr()
        
    #initial estimator 2: spearman correlation matrix
    if ind==2 :
         tmp=np.copy(z)
         y2=tmp
         for j in range(p):
             y2[:,j]=rankdata(tmp[:,j])
         k=pd.DataFrame(y2)
         R=k.corr()
        
    #initial estimator 3: Tukey normal scores
    if ind==3:
         tmp2=np.copy(z)
         y2=tmp2
         for j in range(p):
             y2[:,j]=rankdata(tmp2[:,j])
         y3=norm.ppf((y2-1/3)/(n+1/3))
         k=pd.DataFrame(y3)
         R=k.corr()
         
    #initial estimator 4: spatial sign covariance matrix
    if ind==4:
         for i in range(n):
             if any(z[i,:]==0):
                 z[i,:]=np.tile(0.0001,(1,p))

         
         jerk=np.sqrt(np.sum(z**2,axis=1))
         jerk=jerk.reshape(len(jerk),1)
         k=pd.DataFrame(z/(np.tile(jerk,(1,p))))
         R=k.cov()
        
    #initial estimator 5: BACON
    if ind==5:
        d=np.sqrt(np.sum(z**2,axis=1))
        ind5=np.argsort(d)          #################################################################
        Hinit=ind5[0:int(np.ceil(n/2))]
        k=pd.DataFrame(z[Hinit,:])
        R=k.cov()
   
    #initial estimator 6: Raw OGK estimate for scatter
    if ind==6:
         P,lambda0=og.ogkscatter0(z,scales)
                 
    #calculates initial estimator 

    if ind!=6:
         L,P=np.linalg.eig(R)
         lambda0=np.diagflat(scaler(np.matmul(z,P),scales))**2
     
    sqrtcov=np.matmul(np.matmul(P,fractional_matrix_power(lambda0,(1/2))),P.T)
    covar=np.matmul(np.matmul(P,lambda0),P.T)
    
    sqrtinvcov=np.matmul(np.matmul(P,(fractional_matrix_power(lambda0,(-1/2)))),P.T)
    mu=np.matmul(np.median(np.matmul(z,sqrtinvcov), axis=0),sqrtcov)
         
    dis=rs.resdis0(z,mu,covar)
    disind=np.argsort(dis)
    
    half=round(n/2)
    halfind=disind[:half]
    zhalf=z[halfind,:]
    pnd=pd.DataFrame(zhalf)
    covar=pnd.cov()

    muI=np.mean(zhalf, axis=0)
    scaleI=(np.linalg.det(covar))**(1/(2*p))
    sigmaI=scaleI**(-2)*covar

    
    initest={"mu"    : muI,
             "scale" : scaleI,
             "sigma" : sigmaI
         }
    return initest
 
