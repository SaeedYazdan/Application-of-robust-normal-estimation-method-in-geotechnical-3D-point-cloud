import bc
import TBbdp as tb
import numpy as np
import subsets as sb
import mahalFS as mh
import TBrho as tr
import minscale as ms
import IRWLSmult as IR

print("be notified that the def IRWLSmult is called from another file and this might cause lag.")



"""
def IRWLSmult(Y,initialloc, initialshape, refsteps, reftol, c, kc,initialscale):
    #IRWLSmult (iterative reweighted least squares) does refsteps refining steps from initialloc
    # for refsteps times or till convergence.
    #
    #  Required input arguments:
    #
    #    Y: Data matrix containining n observations on v variables.
    #       Rows of Y represent observations, and columns represent variables.
    # initialloc   : v x 1 vector containing initial estimate of location
    # initialshape: v x v initial estimate of shape matrix
    #   refsteps  : scalar, number of refining (IRLS) steps
    #   reftol    : relative convergence tolerance for the fully iterated
    #               best candidates. Deafult value is 1e-7
    #    c        : scalar, tuning constant of the equation for Tukey biweight
    #   kc        : scalar, tuning constant linked to Tukey's biweight
    #
    #  Optional input arguments:
    #
    # initialscale: scalar, initial estimate of the scale. If not defined,
    #               scaled MAD of Mahalanobis distances from initialloc and
    #               initialshape is used.
    #
    #
    #  Output:
    #
    #  The output consists of a structure 'outIRWLS' containing:
    #      outIRWLS.loc    : v x 1 vector. Estimate of location after refsteps
    #                        refining steps.
    #      outIRWLS.shape  : v x v matrix. Estimate of the shape matrix after
    #                        refsteps refining steps.
    #      outIRWLS.scale  : scalar. Estimate of scale after refsteps refining
    #                        step.
    #      outIRWLS.weights: n x 1 vector. Weights assigned to each observation
    #
    # In the IRWLS procedure the value of loc and the value of the scale and
    # of the shape matrix are updated in each step

    v = size(Y,2)
    loc = initialloc
    # Mahalanobis distances from initialloc and Initialshape
    mahaldist = sqrt(mh.mahalFS0(Y, initialloc, initialshape))


    # The scaled MAD of Mahalanobis distances is the default for the initial scale
    if (nargin < 8):
        initialscale = median(abs(mahaldist))/.6745


    scale = initialscale

    iter = 0
    locdiff = 9999

    while ( (locdiff > reftol) and (iter < refsteps) ):
        iter = iter + 1
    
        # Solve for the scale
        scale = scale* sqrt( mean(tr.TBrho0(mahaldist/scale,c))/kc)
        # mahaldist = vector of Mahalanobis distances from robust centroid and
        # robust shape matrix, which is changed in each step
    
        # compute w = n x 1 vector containing the weights (using TB)
        weights = TBwei(mahaldist/scale,c)
    
        # newloc = new estimate of location using the weights previously found
        # newloc = \sum_{i=1}^n y_i w(d_i) / \sum_{i=1}^n w(d_i)
        newloc = sum(bsxfun(@times,Y,weights),1)/sum(weights)
    
        # exit from the loop if the new loc has singular values. In such a
        # case, any intermediate estimate is not reliable and we can just
        # keep the initial loc and initial scale.
        if (any(isnan(newloc))):
            newloc = initialloc
            newshape = initialshape
            scale = initialscale
            weights=NaN
            break
    
        # Res = n x v matrix which contains deviations from the robust estimate
        # of location
        Res = bsxfun(@minus,Y, newloc)
    
        # Multiplication of newshape by a constant (e.g. v) is unnecessary
        # because final value of newshape remains the same as det(newshape)=1.
        # For the same reason newshape remains the same if we use weights or
        # weights*(c^2/6)
        newshape= (Res')*bsxfun(@times,Res,weights)
        newshape = det(newshape)**(-1/v)*newshape
    
        # Compute MD
        mahaldist = sqrt(mh.mahalFS0(Y,newloc,newshape))
    
        # locdiff is linked to the tolerance
        locdiff = norm(newloc-loc,1)/norm(loc,1)
        loc = newloc
    

    outIRWLS.loc = newloc
    outIRWLS.shape = newshape
    outIRWLS.scale = scale
    outIRWLS.weights=weights

    return outIRWLS


"""



def Smult0(Y, *args):

    

    n=len(Y)
    v=len(Y[0])

    # default value of break down point
    bdpdef=0.5

    # default values of subsamples to extract
    ncomb=bc.bc0(n,v+1)
    nsampdef=min(1000,ncomb)

    # default value of number of refining iterations (C steps) for each extracted subset
    refstepsdef=3;
    # default value of tolerance for the refining steps convergence for  each extracted subset
    reftoldef=1e-6;
    # default value of number of best locs to remember
    bestrdef=5;
    # default value of number of refining iterations (C steps) for best subsets
    refstepsbestrdef=50;
    # default value of tolerance for the refining steps convergence for best subsets
    reftolbestrdef=1e-8;
    # default value of tolerance for finding the minimum value of the scale
    # both for each extracted subset and each of the best subsets
    minsctoldef=1e-7;

    # store default values in the structure options
    options={'nsamp':nsampdef,
            'refsteps':refstepsdef,
            'bestr':bestrdef,
            'reftol':reftoldef,
            'minsctol':minsctoldef,
            'refstepsbestr':refstepsbestrdef,
            'reftolbestr':reftolbestrdef,
            'bdp':bdpdef,
            'plots':0,
            'conflev':0.975,
            'nocheck':0,
            'msg':1,
            'ysave':0}

    #    if nargin > 2:
    #       for i=1:2:length(varargin):
    #           options.(varargin{i})=varargin{i+1}


    bdp = options["bdp"]              # break down point
    refsteps = options["refsteps"]    # refining steps
    bestr = options["bestr"]          # best locs for refining steps till convergence
    nsamp = options["nsamp"]          # subsamples to extract
    reftol = options["reftol"]        # tolerance for refining steps
    minsctol = options["minsctol"]    # tolerance for finding minimum value of the scale for each subset
    msg=options["msg"]                #

    refstepsbestr=options["refstepsbestr"]  # refining steps for the best subsets
    reftolbestr=options["reftolbestr"]      # tolerance for refining steps for the best subsets

    # Find constant c linked to Tukey's biweight
    # rho biweight is strictly increasing on [0 c] and constant on [c \infty)
    # E(\rho) = kc = (c^2/6)*bdp, being kc the K of Rousseeuw and Leroy
    c = tb.TBbdp0(bdp,v)
    kc = ((c**2)/6)*bdp

    
    # Initialize the matrices which contain the best "bestr" estimates of
    # location, index of subsets, shape matrices and scales
    bestlocs = np.zeros((bestr, v))
    bestsubset = np.zeros((bestr, v+1))
    bestshapes = np.zeros((v,v,bestr))
    bestscales = np.inf * np.ones((bestr,1))
    if isinstance(bdp, float):
        lbdp=1
    else:
        lbdp=len(bdp)

    # singsub = scalar which will contain the number of singular subsets which
    # are extracted (that is the subsets of size p which are not full rank)
    singsub=0


    
    print(msg)
    print("+++++++++++++++++++++++++++++++++++++")
    #############################################################################
    ######## Extract in the rows of matrix C the indexes of all required subsets
    C,nselected = sb.subsets0(nsamp,n,v+1,ncomb,msg)
    # Store the indices in varargout
    """
    if nargout==2:
        varargout={C};
    end
    """

    
    # initialise and start timer.
    tsampling = int(np.ceil(min(nselected/100 , 1000)))
    
    time=np.zeros((tsampling,1))
    print(nselected)
    print(C)
    # ij is a scalar used to ensure that the best first bestr non singular
    # subsets are stored
    ij=0
    locj=np.zeros((nselected+1,3))
    maxSize=(nselected+1)*v+1
    Gj=np.zeros((maxSize,3))
    Sj=np.zeros((maxSize,3))
    for i in range( nselected):

        #print(" the number of iteration: "+str(i))
        # if i <= tsampling, tic; end
    
        # find a subset of size v+1 in general position (with rank v).
        index = C[i-1,:]
        #print("index: ")
        #print(index)
        Yj = Y[index-1,:]
        ranky=np.linalg.matrix_rank(Yj)
        #ranky = rank[Yj]
        #print(" and Yj: ")
        #print(Yj)
    
        if ranky==v:
            #print("nselected"+str(nselected))

            #print("Yj")
            #print(Yj)
            #print("mean.Yj")
            #print(np.mean(Yj, axis=0))
            #print(Yj.shape)
            locj[i,:] = np.mean(Yj, axis=0)        # centroid of subset
            #print("iteration: "+str(i))
            
            #print("and the v is: "+str(v))
            #print("cov")
            #print(np.cov(Yj, rowvar=False))
            #print("i*v"+str(i*v)+" and (i+1)*v+1 "+str((i+1)*v+1))
            Sj[i*v:(i+1)*v,:] = np.cov(Yj, rowvar=False)           # covariance of subset
            
            #print("size of Gj"+str(Gj.shape))
            Gj[i*v:(i+1)*v,:] = np.linalg.det(Sj[i*v:(i+1)*v,:])**(-1/v)*Sj[i*v:(i+1)*v,:] # shape of subset

            # Function IRWLSmult performs refsteps steps of IRLS on elemental
            # start. Input:
            # - Y = datamatrix of dimension n x v
            # - locj = row vector containing (robust) centroid
            # - Gj = v x v shape matrix
            # - scale = estimate of the scale (if scale=0). Scaled MAD of Mahalanobis
            #   distances using locj and Gj as centroid and shape matrix is used
            # - refsteps = number of refining iterations
            # - reftol = tolerance for convergence of refining iterations
            # - c = constant of Tukey's biweight linkted to breakdown point
            # - kc = (c^2/6)*bdp
            # Remark: in IRWLSmult the centroid and shape matrix are re-calculated
            # at each step; on the other hand in minscale they are kept fixed. 
        else:
            singsub = singsub + 1

        # if singsub==nsamp
        # error('No subset had full rank. Please increase the number of subsets or check your design matrix X')
        # end
        # if singsub/nsamp>0.1;
        #     disp('------------------------------')
        #     disp(['Warning: Number of subsets without full rank equal to ' num2str(100*singsub/nsamp) '%'])
        # end

    
    for t in range(lbdp):
        sworst = np.inf
        for i in range(nsamp):
            outIRWLS = IR.IRWLSmult0(Y, locj[i,:], Gj[i*v:(i+1)*v,:], refsteps, reftol, c[t], kc[t])
        
            # The output of IRWLSmult is a structure containing centroid, shape
            # matrix and estimate of the scale
            locrw = outIRWLS["loc"]
            shaperw= outIRWLS["shape"]
            scalerw = outIRWLS["scale"]
        
            # Compute Mahalanobis distances using locrw and shaperw
            mdrw = np.sqrt(mh.mahalFS0(Y,locrw,shaperw))
        
            # to find s, save first the best bestr scales and shape matrices
            # (deriving from non singular subsets) and, from iteration bestr+1
            # (associated to another non singular subset), replace the worst scale
            # with a better one as follows
            if ij > bestr:
                # from the second step check whether new loc and new shape belong
                # to the top best loc; if so keep loc and shape with
                # corresponding scale.
                if  np.mean(tr.TBrho0(mdrw/sworst,c[t])) < kc[t]:    # if >kc skip the sample
                    # Find position of the maximum value of bestscale
                    yi = np.argsort(bestscales)
                    ind=yi[bestr-1]
                    print("length of  ind is: "+str(ind))
                    bestscales[ind] = ms.minscale0(mdrw,c[t],kc[t],scalerw,minsctol)
                    bestlocs[ind,:] = locrw
                    bestshapes[:,:,ind] = shaperw
                    sworst = max(bestscales)
                    # best subset sssociated with minimum value
                    # of the scale
                    bestsubset[ind,:]=index
            else:
                print("length of  sth is: "+str(len(bestscales)))
                print("ij is: "+str(ij))
                bestscales[ij-1] = ms.minscale0(mdrw,c[t],kc[t],scalerw,minsctol)
                bestlocs[ij-1,:] = locrw
                bestshapes[:,:,ij-1] = shaperw
                bestsubset[ij-1,:] = index
                ij=ij+1
    
        # Write total estimate time to compute final estimate
        # if i <= tsampling
        
            # sampling time until step tsampling
         #   time(i)=toc;
        # elseif i==tsampling+1
            # stop sampling and print the estimated time
         #   if msg==1
         #       fprintf('Total estimated time to complete S estimator: %5.2f seconds \n', nselected*median(time));
         #   end
        #end
    
    # perform C-steps on best 'bestr' solutions, till convergence or for a
    # maximum of refstepsbestr steps using a convergence tolerance as specified
    # by scalar reftolbestr

    # this is to ensure that the condition tmp.scale < superbestscale in the
    # next if statement is satisfied at least once
    superbestscale = np.inf
    for i in range(bestr):
        tmp = IR.IRWLSmult0(Y,bestlocs[i,:], bestshapes[:,:,i],refstepsbestr,reftolbestr,c[t],kc[t], bestscales[i])
        if tmp["scale"] < superbestscale:
            superbestscale  = tmp["scale"]
            superbestloc     = tmp["loc"]
            superbestshape  = tmp["shape"]
            superbestsubset = bestsubset[i,:]
            weights = tmp["weights"]

            

    out={"class": 'S',
         "loc": superbestloc,
         "shape": superbestshape,
         "objective": superbestscale**(2*v),
         "cov": superbestscale**2*superbestshape,
         "weights": weights,
         "bs": superbestsubset,
         "c": C
        }

    
    """
    out.class   = 'S';
    out.loc[t,:]     = superbestloc         # robust estimates of location
    out.shape((t-1)*v+1:t*v,:)   = superbestshape       # robust estimates of shape matrix
    out.scale[t]   = superbestscale       # robust estimates of the scale
    out.objective[t]=superbestscale**(2*v)
    out.cov((t-1)*v+1:t*v,:)     = superbestscale**2*superbestshape #robust estimates of covariance matrix
    out.weights = weights
    #out.md(t) = mahalFS(Y,out.loc(t,:),out.cov((t-1)*v+1:t*v,:));
    out.bs=superbestsubset             # Store units formin best subset
    out.C=C
    """

    
    return out




