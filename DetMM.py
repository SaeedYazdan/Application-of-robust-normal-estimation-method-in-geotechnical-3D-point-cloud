import numpy as np
import time

import DetS as ds
import MMmultcore as mm

def DetMM0(Y,*args, **kwargs):

    ## Beginning of code
    inpu=locals()
    arug=inpu["args"]
    karug=inpu["kwargs"]

    n=len(Y)
    v=len(Y[0])

    # default values for the initial S estimate:

    # default value of break down point
    Sbdpdef=0.5
    # default values of subsamples to extract
    Snsampdef=20
    # default value of number of refining iterations (C steps) for each extracted subset
    Srefstepsdef=3
    # default value of tolerance for the refining steps convergence for  each extracted subset
    Sreftoldef=1e-6
    # default value of number of best locs to remember
    Sbestrdef=5
    # default value of number of refining iterations (C steps) for best subsets
    Srefstepsbestrdef=50
    # default value of tolerance for the refining steps convergence for best subsets
    Sreftolbestrdef=1e-8
    # default value of tolerance for finding the minimum value of the scale 
    # both for each extracted subset and each of the best subsets
    Sminsctoldef=1e-7      

    options={'InitialEst':'',
             'Snsamp':      Snsampdef,
             'Srefsteps':   Srefstepsdef,
             'Sbestr':      Sbestrdef,
             'Sreftol':     Sreftoldef,
             'Sminsctol':   Sminsctoldef,
             'Srefstepsbestr':Srefstepsbestrdef,
             'Sreftolbestr':Sreftolbestrdef,
             'Sbdp':        Sbdpdef,
             'nocheck':     0,
             'eff':         0.95,
             'effshape':    0,
             'refsteps':    100,
             'tol':         1e-7,
             'conflev':     0.975,
             'plots':       0,
             'ysave':       0}

    if len(karug)>0:
        for keys in arugk:
            options[keys]=karug[keys]

            

    out={"loc" :        0,
         #"shape" :      0,
         "cov" :        0,
         #"Scov" :       0,
         #"Sloc" :       0,
         #"Sshape" :     0,
         #"auxscale" :   0,
         #"weights" :    0,
         "outliers" :   0,
         "conflev" :    0
         #"class" :      " "
        }

    InitialEst=options["InitialEst"]

    #if isempty(InitialEst):
        
        # first compute S-estimator with a fixed breakdown point
    bdp = options["Sbdp"]
    Sresult, rub=ds.DetS0(Y,bdp)
        
    auxscale = Sresult["scale"]
    shape0 = Sresult["shape"]
    loc0 = Sresult["mean"]
    #else:
        #auxscale = InitialEst.auxscale
        #shape0 = InitialEst.shape0
        #loc0 = InitialEst.loc0


    ## MM part

    # Asymptotic nominal efficiency (for location or shape)
    eff=options["eff"]

    # effshape = scalar which specifies whether nominal efficiency refers to
    # location or scale
    effshape=options["effshape"]

    # refsteps = maximum number of iterations in the MM step
    refsteps=options["refsteps"]

    # tol = tolerance to declare convergence in the MM step
    tol = options["tol"]
    conflev=options["conflev"]

    # MMmultIRW = function which does IRWLS steps from initial loc (loc0) and
    # initial shape matrix (Shape0). The estimate of sigma (auxscale) remains
    # fixed inside this routine

    try:
        lbdp=len(bdp)
    except:
        lbdp=1


    locM     = np.zeros((lbdp,v))
    #shapeM   = np.zeros((lbdp*v,v))
    covM     = np.zeros((lbdp*v,v))
    #ScovM    = np.zeros((lbdp*v,v))
    
    for t in range(lbdp):
        outIRW = mm.MMmultcore0(Y,loc0[t,:], shape0[t*v:(t+1)*v,:] ,auxscale[t], eff=eff, effshape=effshape, refsteps=refsteps, reftol=tol, conflev=conflev);
        locM[t,:]      = outIRW["loc"]
        #shapeM[t*v:(t+1)*v,:]   = outIRW["shape"]
        covM[t*v:(t+1)*v,:]     = outIRW["cov"]
        #ScovM[t*v:(t+1)*v,:]    = auxscale[t]**2*shape0[t*v:(t+1)*v,:]
        
    # --------------------------------------------------------------------

    out["loc"]     = locM
    #out["shape"]     = shapeM
    out["cov"]     = covM
    #out["Scov"]     = ScovM
    #out["Sloc"]     = loc0
    #out["Sshape"]   = shape0
    #out["auxscale"] = auxscale

    #out["weights"] = outIRW["weights"]
    out["outliers"]= outIRW["outliers"]
    out["conflev"] = outIRW["conflev"]
    #out.md = mahalFS(Y,out.loc,out.cov)  % store md (in squared units)
    #out["class"]   = 'MM'

    plo=options["plots"]


    """

    # Plot Mahalanobis distances with outliers highlighted
    if (isstruct(plo) or (~isstruct(plo) and plo!=0)):
        
        [n,v]=size(Y)
        
        laby='MM Mahalanobis distances'
        malindexplot(out.md,v,'conflev',conflev,'laby',laby,'numlab',out.outliers)
        
        figure('Tag','pl_spm_outliers')
        group=ones(n,1)
        if ~isempty(out["outliers"]):
            group(out["outliers"])=2

        spmplot(Y,group,plo)
        set(gcf,'Name',' MM estimator: scatter plot matrix with outliers highlighted')


    if options["ysave"]:
        out["Y"] = Y
    """
    return out

