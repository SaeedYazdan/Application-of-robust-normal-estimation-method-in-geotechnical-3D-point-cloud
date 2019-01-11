import numpy as np

import qn
import DetS as ds

def ogkscatter0(Y,scales):
    n=len(Y)
    p=len(Y[0])
    U=np.eye(p)
    for i in range(p):
        sYi=Y[:,i]
        for j in range(i):
            sYj=Y[:,j]
            sY=sYi+sYj
            dY=sYi-sYj
            if scales=='qn':
                temp=qn.qn0(sY)
                temp0=qn.qn0(dY)
            else:
                temp=ds.W_scale(sY)
                temp0=ds.W_scale(dY)
                
            U[i,j]= 0.25*(temp**2-temp0**2)

    U=np.tril(U,-1)+U.T
    L,P=np.linalg.eig(U)
    Z=np.matmul(Y,P)
    if scales=='qn':
        sigz=qn.qn0(Z)
    else:
        sigz=ds.W_scale(z)

    lambda0=np.diagflat(sigz**2)
    #P[:,[0,2]] = P[:,[2,0]]
    #lambda0[:,[0,2]] = lambda0[:,[2,0]]
    #lambda0[[0,2],:] = lambda0[[2,0],:]
    return P, lambda0
