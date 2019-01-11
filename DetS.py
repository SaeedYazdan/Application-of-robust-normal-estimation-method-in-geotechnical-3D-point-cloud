import numpy as np
import time

import Tbsc as ts
import Tbsb as tb
import qn
import initnew as iw
import res
import resdis as rd
import scalefull as sc
import lossS as ls


def DetS0(x,bdp):

    
    ## Beginning of the code

    scale_est=1
    n=len(x)
    p=len(x[0])

    
    if n>1000:
        scales='W_scale'
    else:
        scales='qn'


    # Some checks are now performed.
    if n==0:
        error('All observations have missing or infinite values.')

    if n < p:
        error('Need at least (number of variables) observations.')

    
    #setting the parameters
    bestr=2
    numbinit=6
    k=2
    try:
        lbdp=len(bdp)
    except:
        lbdp=1
        
    c = ts.Tbsc0(bdp,p)
    kp = (c/6)* tb.Tbsb0(c,p)
    
    mu = np.zeros((numbinit,p))
    murw = np.zeros((bestr,p))
    sigma = np.zeros((numbinit*p,p))
    sigmarw = np.zeros((bestr*p,p))
    rdisrw=np.zeros((bestr,n))
    scalerw=1e20*np.ones((1,bestr))
    bestscales = 1e20*np.ones((bestr))
    bestindices=np.zeros((bestr,1))
    bestmus=np.zeros((bestr,p))
    bestsigmas=np.zeros((numbinit,p))


    meanR=np.zeros((1,p))
    covarianceR=np.zeros((p,p))
    detR=np.zeros((1))
    scaleR=np.zeros((1))
    shapeR=np.zeros((p,p))
    bestindexR=np.zeros((1))
    objectiveR=np.zeros((1))
    

    #scale the data
    med=np.median(x, axis=0)
    cx=x-np.tile(med,(n,1))
    if scales=='qn':
        sca=qn.qn0(x)
    elif scales=='W_scale':
        sca=W_scale(x)
    
    z=cx/np.tile(sca,(n,1))

    #Determining initial location and shape estimates
    for i in range(numbinit):
        #i=1 Hyperbolic tangent of standardized data
        #i=2 Spearman correlation matrix
        #i=3 Tukey normal scores
        #i=4 spatial sign covariance matrix
        #i=5 BACON
        #i=6 Raw OGK estimate for scatter
        initest=iw.initnew0(z,i+1,scales)
        mu[i,:]=initest["mu"]
        sigma[i*p:((i+1)*p),:]=initest["sigma"]

    for j in range(lbdp) :
        sworst = 1e20;
        for i in range(numbinit) :
                
                if k>0  :  #Refine using 2 I-steps to determine the best scale
                    tmp=res.res0(z,mu[i,:],sigma[i*p:(i+1)*p,:],k,0,kp[j],c[j])                    
                    murw = tmp["murw"]
                    sigmarw = tmp["sigmarw"]
                    scalerw = tmp["scalerw"]
                    rdisrw = rd.resdis0(z,murw,sigmarw)
                    
                    #print("sigmarw"+str(sigmarw))
                else:
                    murw = mu
                    sigmarw = sigma
                    rdisrw = resdis(z,murw,sigmarw)
                    scalerw = median(abs(rdisrw))/.6745

                
                if i > 0 :   # if this isn't the first iteration....
                     scaletest = ls.lossS0(rdisrw,sworst,c[j])                
                     
                     if scaletest < kp[j] :
                         sbest = sc.scalefull0(rdisrw,kp[j],c[j],scalerw)
                         yi=np.argsort(bestscales)
                         ind=yi[bestr-1]
                         bestscales[ind] = sbest
                         bestmus[ind,:] = murw
                         bm1 = (ind)*p
                         bestsigmas[(bm1):(bm1+p),:] = sigmarw
                         sworst = (bestscales).max()
                         bestindices[ind]=i+1


                else   :    # if this is the first iteration, then this is the best beta...
                    bestscales[bestr-1] = sc.scalefull0(rdisrw,kp[j],c[j],scalerw)                    
                    bestmus[bestr-1,:] = murw
                    bm1 = (bestr-1)*p
                    bestsigmas[(bm1):(bm1+p),:] = sigmarw
                    bestindices[bestr-1]=i+1

        # do the complete refining step until convergence (conv=1) starting
        # from the best subsampling candidate (possibly refined)
        superbestscale = 1e20

        # magic number alert
        for i in range(bestr,0,-1) :

            index = (i-1)*p
            tmp = res.res0(z,bestmus[i-1,:],bestsigmas[(index):(index+p),:],0,1,kp[j],c[j],bestscales[i-1])

                
            if tmp["scalerw"] < superbestscale :
                superbestscale = tmp["scalerw"]
                superbestmu = tmp["murw"]
                superbestsigma= tmp["sigmarw"]
                #superbestindex=bestindices[i-1]

        
        #bestcov=superbestsigma*superbestscale**2
        
        meanR[j,:]=superbestmu*sca+med
        #covarianceR[j*p:(j+1)*p,:]=bestcov*np.tile(sca,(p,1))*np.tile(sca.T,(1,p))        
        #detR[j]=np.linalg.det(covarianceR[j*p:(j+1)*p,:])
        Q=np.diagflat(sca)
        scaleR[j]=superbestscale*np.linalg.det(Q)**(1/p)
        
        shapeR[j*p:(j+1)*p,:]=np.linalg.det(Q)**(-2/p)*superbestsigma*np.tile(sca,(p,1))*np.tile(sca.T,(1,p))
        #bestindexR[j]=superbestindex
        #objectiveR[j]=superbestscale**(2*p)*(np.linalg.det(Q))**2

        
        # rescale to the original data
        result={
            "mean" : meanR,
            #"covariance" : covarianceR,
            #"det" : detR,
            "scale" : scaleR,
            "shape" : shapeR,
            #"bestindex" : bestindexR,
            #"objective" : objectiveR
            }


    return result,z

"""                                                                                        

    #--------------------------------------------------------------------------
def W_scale(x):
    c=4.5
    [n,p]=size(x)
    Wc=inline('(1-(x./c).^2).^2.*(abs(x)<c)')
    sigma0=mad(x,1)
    w=Wc(c,(x-repmat(median(x),n,1))/repmat(sigma0,n,1))
    loc=diag(x.T*w).T/sum(w)
    c=3
    rc=inline('min(x**2,c**2)')
    sigma0=mad(x,1)
    b=c*norminv(3/4)
    nes=n*(2*((1-b**2)*normcdf(b)-b*normpdf(b)+b**2)-1)
    scale=sigma0**2/nes*sum(rc(c,(x-repmat(loc,n,1))/repmat(sigma0,n,1)))
    scale=sqrt(scale)
    return scale
"""
    
