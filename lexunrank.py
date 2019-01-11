import numpy as np
import bc

def lexunrank0(n,k,N,*arg):
    inpu=locals()
    arug=inpu['arg']
    #print(arug)
    kcomb=np.zeros((1,k))
    calls =0

    
    if len(arug)<1:
        N_kk=N
        pas_col=np.ones((n,1))
        seq=np.arange(1,n+1)
        for kk in range(k,0,-1):

            # The next 'if' statement builds the required part of column kk+1
            # of the pascal matrix, which is the argument of the 'find'
            # statement which follows.
            # This replaces the loop with repeated calling of bc:
            #         for x = kk:n-1
            #             if  bc(x,kk)> N_kk, break, end
            #         end

            if kk == k:
                for x2 in range(kk,n):
                    pas_col[x2] = pas_col[x2-1]*(x2+1)/(x2+1-kk)
            else:
                divi=np.array(seq[kk:n+1]-kk)[np.newaxis]
                
                #inve=np.linalg.inv(seq[kk:n+1]-kk)
                #pas_col[kk+1:n+1] = pas_col[kk+1:n+1]*(kk+1)*inve
                pas_col[kk:n+1] = np.divide(pas_col[kk:n+1]*(kk+1),divi.T)


            #x = find(pas_col[kk:end] > N_kk,1)
            x=np.where(pas_col[kk-1:] > N_kk)
        
            if x[0].size==0:
                maxx=n-1
                calls=calls+maxx-kk+1
                
            else:
                maxx = x[0][0]+kk-1
                calls=calls+maxx-kk+2

            
            kcomb[0,kk-1]=n-maxx  ###################
            if maxx>=kk:
                N_kk = N_kk - bc.bc0(maxx,kk)
                calls = calls+1
    else:

        pascalM=arug[0]
        ## FAST OPTION:
        # binomial coefficients are taken from the pascal matrix rather than
        # computing them using bc. Of course this option is space greedy.

        N_kk = N
    
        for kk in range(k,0,-1):

            #x=find(pascalM[1:n-kk,kk+1] > N_kk , 1)
            x=np.where(pascalM[0:n-kk,kk] > N_kk)
        
            if x[0].size==0: # || x1==n-kk
                maxx=n-1
                calls=calls+maxx-kk+1
            else:
                maxx = x[0][0]+kk-1
                calls=calls+maxx-kk+2
            kcomb[0,kk-1]=n-maxx

            if maxx >= kk:
                N_kk = N_kk - pascalM[maxx-kk,kk]  # this is: N_kk - bc(maxx,kk)
                calls = calls+1



    return kcomb, calls
