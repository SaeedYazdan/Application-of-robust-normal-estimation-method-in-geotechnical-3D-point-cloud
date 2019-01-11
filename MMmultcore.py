import numpy as np
from scipy import stats as st

from TBeff import TBeff0
import TBrho as tr
import TBwei as tw
import mahalFS as mh


def example0():
    import Smult as sm
    y=np.zeros((200,2))

    for i in range(len(y)):
        y[i,0]=np.random.uniform(-10,10)
        y[i,1]=np.random.uniform(-1,1)


    return sm.Smult0(y,'plots',1)

#############################################################################################




def MMmultcore0(Ycont,loc0, shape0, auxscale,**kwargs):
    # Here we go, the function starts here

    inpu=locals()
    
    arug=inpu["kwargs"]
    
    ################# checking data
    rank=np.linalg.matrix_rank(Ycont)


    if rank < len(Ycont[0]):
        print("Matrix X is singular")
        exit()
    #
    #
    #
    #
    #################################

    num_of_inp=8
    n=len(Ycont)
    v=len(Ycont[0])


    # default nominal efficiency
    effdef = 0.95
    # by default the nominal efficiency refers to location efficiency
    effshapedef = 0
    # default value of number of maximum refining iterations
    refstepsdef=50
    # default value of tolerance for the refining steps convergence
    reftoldef=1e-6


    # store default values in the structure options
    """
    options=struct('refsteps',refstepsdef,'reftol',reftoldef,
        'eff',effdef,'effshape',effshapedef,'conflev',0.975,
        'plots',0,'nocheck',0,'ysave',0)
    """
    options={'refsteps':refstepsdef,
             'reftol':reftoldef,
             'eff':effdef,
             'effshape':effshapedef,
             'conflev':0.975,
             'plots':0,
             'nocheck':0,
             'ysave':0}



    eff     = options["eff"]      # nominal efficiency
    effshape= options["effshape"] # nominal efficiency refers to shape or location
    refsteps= options["refsteps"] # maximum refining iterations
    reftol  = options["reftol"]   # tolerance for refining iterations covergence

    if len(arug)>0:
        for keys in arug:
            options[keys]=arug[keys]
        
    if effshape==1:
        c = TBeff0(eff,v,1)
    else:
        c = TBeff0(eff,v)

    # Ytilde = deviations from centroid

    Ytilde = Ycont- loc0
    invShape=np.linalg.inv(shape0)
    divi=np.matmul(Ytilde,invShape)
    # mahaldist = vector of Mahalanobis distances using as shape matrix shape0
    mahaldist = np.sqrt(np.sum(divi*Ytilde,axis=1))

    # newobj = objective function which has to be minimized (keeping the value
    # of the scale fixed)
    origobj  = np.mean(tr.TBrho0(mahaldist/auxscale,c))
    newobj   = origobj

    # compute M-estimate with auxiliary scale through IRLS steps, starting
    # from S-estimate
    iter0   = 0
    oldobj = newobj + 1

    while ((oldobj - newobj) > reftol) and (iter0 < refsteps):
        iter0 = iter0 +1
        
        # w = psi(x)/x
        weights = tw.TBwei0(mahaldist/auxscale,c)
        weights=weights.reshape(len(weights),1)
        
        # Find new estimate of location using the weights previously found
        # newloc = \sum_{i=1}^n y_i weights(d_i) / \sum_{i=1}^n weights(d_i)
        newloc = np.sum(Ycont*weights,axis=0)/np.sum(weights)       
        # exit from the loop if the new loc has singular values. In such a
        # case, any intermediate estimate is not reliable and we can just
        # keep the initial loc and initial scale.
        if (any(np.isnan(newloc))):
            newloc = loc0
            #newshape = shape0
            weights=np.nan
            break

        # Find new estimate of scaled covariance matrix using  the weights
        # previously found
        newshape = np.matmul((Ytilde*weights).T,Ytilde)
    
        # newshape is a var cov matrix with determinant equal to 1
        newshape = np.linalg.det(newshape)**(-1/v)*newshape
   
        # Compute Mahalanobis distances from centroid newloc and var newshape
        mahaldist=np.sqrt(mh.mahalFS0(Ycont,newloc,newshape))
                          
        oldobj = newobj
        newobj = np.mean(tr.TBrho0(mahaldist/auxscale,c))        

    if newobj <= origobj:
        loc1 = newloc
        #shape1 = newshape
        cov1 = auxscale**2*newshape
        #weights1 = weights
        
    else:        # isn't supposed to happen
        raise NameError('Initial solutions for location and shape parameters have been kept')
        raise NameError('Because MM-loop does not produce better estimates');
        loc1 = loc0
        #shape1 = shape0
        cov1 = auxscale**2*shape0
        #weights1 = np.NaN
        
    md0 = mh.mahalFS0(Ycont,loc1,cov1)

    # Store in output structure the outliers found with confidence level conflev
    conflev = options["conflev"]
    seq = np.arange(n)
    outliers1 = seq[md0 > st.chi2.ppf(conflev,v) ]


    out={#"class" : 'MM',
         "loc" :  loc1,
         #"shape":  shape1,
         "cov" :  cov1,
         #"weights" :  weights1,
         #"md" : md0,
         "outliers" :  outliers1,
         "conflev" :  conflev,
         "Y" : 0
        }
        

    if options["ysave"]:
        out["Y"] = Y

    #plo=options["plots"]
    return out
    
"""
    # Plot Mahalanobis distances with outliers highlighted
    if isstruct(plo) or (~isstruct(plo) and plo!=0):
        
        [n,v]=size(Y)
        
        laby='MM Mahalanobis distances'
        malindexplot(out.md,v,'conflev',conflev,'laby',laby,'numlab',out.outliers)
        
        figure('Tag','pl_spm_outliers')
        group=ones(n,1)
        if isempty(out["outliers"]):
            #group(out["outliers"])=2
            print("kos")
        spmplot(Y,group,plo)
        set(gcf,'Name',' MM estimator: scatter plot matrix with outliers highlighted')   
"""
    

