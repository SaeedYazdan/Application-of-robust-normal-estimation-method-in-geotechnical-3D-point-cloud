import mahalFS as mh
import TBwei as tw
import TBrho as tr
import numpy as np

def IRWLSmult0(Y,initialloc, initialshape, refsteps, reftol, c, kc,*args):

    inpu=locals()
    arug=inpu["args"]
    v = len(Y[0])
    loc = initialloc
    print("initialshape: ")
    print(initialshape)
    print("Y: ")
    print(Y)
    print("Mahal ")
    print(mh.mahalFS0(Y, initialloc, initialshape))
    # Mahalanobis distances from initialloc and Initialshape
    mahaldist = np.sqrt(mh.mahalFS0(Y, initialloc, initialshape))


    # The scaled MAD of Mahalanobis distances is the default for the initial scale

    if len(arug)<1:
        initialscale = np.median(abs(mahaldist))/.6745
    else:
        initialscale=arug[0]

    scale = initialscale

    iter = 0
    locdiff = 9999

    while ( (locdiff > reftol) and (iter < refsteps) ):
        iter = iter + 1
    
        # Solve for the scale
        scale = scale* np.sqrt( np.mean(tr.TBrho0(mahaldist/scale,c))/kc)
        # mahaldist = vector of Mahalanobis distances from robust centroid and
        # robust shape matrix, which is changed in each step
    
        # compute w = n x 1 vector containing the weights (using TB)
        weights = tw.TBwei0(mahaldist/scale,c)
    
        # newloc = new estimate of location using the weights previously found
        # newloc = \sum_{i=1}^n y_i w(d_i) / \sum_{i=1}^n w(d_i)
        print("Y")
        print(Y)
        print("weights")
        print(weights)
        weights1=weights.reshape(len(weights),1)
        weights=weights1
        newloc = sum(np.multiply(Y,weights))/sum(weights)    ###############################
    
        # exit from the loop if the new loc has singular values. In such a
        # case, any intermediate estimate is not reliable and we can just
        # keep the initial loc and initial scale.
        print("is nan: ")
        print(np.isnan(newloc))
        if (any(np.isnan(newloc))):
            newloc = initialloc
            newshape = initialshape
            scale = initialscale
            weights=NaN
            break
    
        # Res = n x v matrix which contains deviations from the robust estimate
        # of location
        Res = Y- newloc
    
        # Multiplication of newshape by a constant (e.g. v) is unnecessary
        # because final value of newshape remains the same as det(newshape)=1.
        # For the same reason newshape remains the same if we use weights or
        # weights*(c^2/6)
        newshape= np.dot((Res.T),(np.multiply(Res,weights)))
        print("det is computed")
        print(len(np.multiply(Res,weights)))
        newshape = np.linalg.det(newshape)**(-1/v)*newshape
    
        # Compute MD
        mahaldist = np.sqrt(mh.mahalFS0(Y,newloc,newshape))
    
        # locdiff is linked to the tolerance
        locdiff = np.linalg.norm(newloc-loc,1)/np.linalg.norm(loc,1)
        loc = newloc


    outIRLWS={"loc": newloc,
              "shape": newshape,
              "scale": scale,
              "weights": weights
        }

    return outIRLWS
    
