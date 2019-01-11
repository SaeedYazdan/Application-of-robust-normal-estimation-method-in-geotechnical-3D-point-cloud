import numpy as np
from scipy import spatial as sp

def Qncore(x):
        
    n=len(x)
    # Do binning for big n
    if n>10000:
        sy=sort(x)
        nbins=np.floor(n/10)
        xbinned=np.zeros(nbins,1)
        ninbins=np.floor(n/nbins)
        for ii in range(nbins):
            if (mod(n,nbins)!=0 and ii==nbins):
                xbinned[ii]=median(sy[(ii-1)*ninbins+1:n])
            else:
                xbinned[ii]=median(sy[(ii-1)*ninbins+1:ii*ninbins])

        x=xbinned                  # Redefine x with binned x
        n=nbins                    # Redefine n with number of bins

    h=np.floor(n/2)+1
    kk=0.5*h*(h-1)
        
    # Compute the n*(n-1)/2 pairwise ordered distances
    # Use function pdist of statistics toolbox
    x=x.reshape(len(x),1)
    
    distord=np.sort(sp.distance.pdist(x,metric='cityblock'))
        
    #        If statistic toolbox is not present it is possible to use the following code
    #         distord = zeros(1,n*(n-1)./2);
    #         kkk = 1;
    #         for iii = 1:n-1
    #             d = abs(x(iii) - x((iii+1):n));
    #             distord(kkk:(kkk+n-iii-1)) = d;
    #             kkk = kkk + (n-iii);
    #         end
    #         distord=sort(distord);

    
    s=2.2219*(distord[int(kk)-1])
    # Multiply the estimator also by cn a finite sample correction
    # factor to make the estimator unbiased for finite samples (see p. 10
    # of Croux and Rousseeuw, 1992) or
    # http://www.google.it/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&ved=0CDUQFjAA&url=http%3A%2F%2Fwww.researchgate.net%2Fpublication%2F228595593_Time-efficient_algorithms_for_two_highly_robust_estimators_of_scale%2Ffile%2F79e4150f52c2fcabb0.pdf&ei=ZCE5U_qHIqjU4QTMuIHwAQ&usg=AFQjCNERh4HiLgtkUGF1w4JU1380xhvKhA&bvm=bv.63808443,d.bGE

    options = {
           2 : 0.399,
           3 : 0.994,
           4 : 0.512,
           5 : 0.844,
           6 : 0.611,
           7 : 0.857,
           8 : 0.669,
           9 : 0.872
    }
    
    if n==1:
        print("Sample size too small")
    elif n>9:
        if n%2==1:
            dn=n/(n+1.4)
        elif n%2==0:
            dn=n/(n+3.8)
    else:
        dn=options[n]


    s=dn*s
    return s


#########################################################################################################################


def qn0(X, *args):
    inpu=locals()
    arug=inpu['args']
    #dim=arug[0]
    ndim=X.shape
    ndimA=np.array(ndim)

    try:
        leng=len(arug)
    except:
        leng=0

    try:
        ndoom=ndimA[1]
    except:
        ndoom=0

        
    ## Beginning of code

    if (leng == 0 and ndoom==0):   # Input is a column  vector
        y=Qncore(X)
    elif (leng == 0 and ndimA[0]==0) :   #Input is a row vector
        y=Qncore(X.T)
    else :                                 # Input is at least a two dimensional array
        if (leng == 0):               # Determine first nonsingleton dimension
            #dim = find(ndim!=1,1)
            dim=np.where(ndim!=1)[0][0]+1
        else:
            dim=arug[0]

        if dim > len(ndim):                # If dimension is too high, just return input.
            return X

        if len(ndim)==2:

            if dim==1 :                 # Input is a matrix  dim=1 compute along columns
                y=np.zeros((1,ndim[1]))
                for j in range(ndim[1]) :
                    y[:,j]=Qncore(X[:,j])

            else:                       # Input is a matrix  dim=2 compute along rows
                y=np.zeros((ndim[0],1))
                
                for i in range(ndim[0]):
                    y[i]=Qncore(X[i,:].T)

        elif len(ndim)==3:
            if dim==1:                  # Input is a 3D array dim=1 compute along columns
                y=np.zeros((1,ndim[1],ndim[2]));
                for k in range(ndim[2]) :
                    for j in range(ndim[1]):
                        y[1,j,k]=Qncore(X[:,j,k])

            
            elif dim==2 :               # Input is a 3D array dim=2 compute along rows
                y=np.zeros((ndim[0],1,ndim[2]))
                for k in range(ndim[2]):
                    for i in range(ndim[0]):
                        y[i,1,k]=Qncore(X[i,:,k].T)

            
            else :                      # Input is a 3D array dim=3 compute along 3rd dim
                y=np.zeros((ndim[0],ndim[1]))
            
                for i in range(ndim[0]) :
                    for j in range(ndim[1]) :
                        y[i,j]=Qncore(X[i,j,:].T)

        else:
            raise NameError('FSDA:Qn:WrongInput','Not implemented for array of size greater than 3')

    return y
