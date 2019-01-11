import numpy as np
import bc
import combsFS as cs
import randsampleFS as rs
import lexunrank as lx
import platform
from scipy.linalg import pascal
import ctypes



class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]

    def __init__(self):
        # have to initialize this to the size of MEMORYSTATUSEX
        self.dwLength = ctypes.sizeof(self)
        super(MEMORYSTATUSEX, self).__init__()




def subsets0(nsamp,n,p,*args):
    inpu=locals()
    inp=len(inpu)
    arug=inpu['arg']

    if len(arug)<1:
        ncomb=bc.bc0(n,p)
    else:
        ncomb=arug[0]

    if len(arug)<2:
        msg=1
    else:
        msg=arug[1]

    seq=np.arange(1,n+1)

    ######################################################    
    ## Combinatorial part to extract the subsamples
    # Key combinatorial variables used:
    # C = matrix containing the indexes of the subsets (subsamples) of size p
    # nselected = size(C,1), the number of all selected subsamples.
    # nselected = number of combinations on which the procedure is run.
    # rndsi = vector of nselected indexes randomly chosen between 1 e ncomb.
    Tcomb = 5e+7
    T2comb = 1e+8
    print("ncomb: "+str(ncomb)+" Tcomb: "+str(Tcomb))
   
    
    if (nsamp==0 or ncomb <= Tcomb):
        if nsamp==0:
            if (ncomb > 100000 and msg==1):
                print('Warning: you have specified to extract all subsets (ncomb='+str(ncomb)+')')
                print('The problem is combinatorial and the run may be very lengthy')
            nselected = ncomb
        else:
            nselected = nsamp

        # If nsamp = 0 matrix C contains the indexes of all possible subsets
        print("+++++++++++++++++++++++++++++++++++++**************************")
        
        C=cs.combsFS0(seq,p)
        #print(C)
    
        # If nsamp is > 0 just select randomly ncomb rows from matrix C
        if nsamp>0:
            print("ncomb: "+str(ncomb))
            print("nsamp: "+str(nsamp))
            # Extract without replacement nsamp elements from ncomb
            rndsi=rs.randsampleFS0(ncomb,nsamp,2)
            #print(rndsi)
            print("gooooooooooooooooooooooooooooooooooz")
            C = C[rndsi-1,:] ###########################################
            #print(C.shape)
            #print(C)
        #end
    else:

        print(" the codes have not been checked yet, they might be funny")

        
        if (nsamp > 100000 and msg==1):
            print('Warning: you have specified to extract more than 100000 subsets')
            print('To iterate so many times may be lengthy')
        nselected = nsamp
        
        usepascal=0

        if (ncomb>Tcomb and ncomb<T2comb):
        
            # Extract without replacement nsamp elements from ncomb
            rndsi=rs.randsampleFS0(ncomb,nsamp,2)
        
            if platform.system()=='Windows':
                print("yoooohoooo")


                stat = MEMORYSTATUSEX()
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
                print(stat.ullAvailPhys)



                
                #[~,sys]=memory
                bytesavailable=stat.ullAvailPhys
                if bytesavailable > 2*8*n**2:
                    pascalM=pascal(n)
                else:
                    pascalM=pascal(n)
                usepascal=1
            

        if n < 2**15:
            C=np.zeros((nselected,p),'int16')
        else:
            C=np.zeros((nselected,p),'int32')


        print(rndsi)

        for i in range(1,nselected+1):
            
            if (ncomb>Tcomb and ncomb<T2comb):
            
                if usepascal:
                    #print(" here n is: "+str(n))
                    #print(" here p is: "+str(p))
                    #print(" here rndsi[i] is: "+str(rndsi[i]))
                    #print(" here pascalM is: "+str(pascalM)) 
                    s, calls=lx.lexunrank0(n,p,rndsi[i-1],pascalM)
                    #print("sssssssssssssssssss")
                    #print(s)
                else:
                    s, calls=lx.lexunrank0(n,p,rndsi[i-1])

            else:
                
                s, calls=rs.randsampleFS0(n,p)

            C[i-1,:]=s
        


    
    return C, nselected
