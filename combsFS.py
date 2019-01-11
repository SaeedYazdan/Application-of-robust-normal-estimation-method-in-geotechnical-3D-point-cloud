import numpy as np
import bc

def combsFS0(v,m):
    inp=len(locals())
    ## Beginning of code
    if inp !=2:
        print('FSDAtoolbox:combsFS:WrongInputNum'+ 'Requires 2 input arguments.')
        exit()

    v = np.transpose(v)     # Make sure v is a row vector.
    n = len(v)  # Elements of v.

    # set the *theoretical* precision based on the number of elements in v.
    # Of course so big n values will never be used in practice. The precision
    # will always be int8.
    if n < 128:
        precision = 'int8'
        v = np.int8(v)
    elif n < 32768:
        precision = 'int16'
        v = np.int16(v)
    else:
        precision = 'int32'
        v = np.int32(v)

    
    if(m > n):
        print('FStoolbox:combsFS:WrongInputNum: '+ 'm > n !!');
    elif n == m:
        P = v
    elif m == 1:
        P = np.transpose(v)
    elif(m == 0):
        P=[]
    else:
       #The binomial coefficient (n choose m) can be computed using
       #prod(np1-m:n)/prod(1:m). For large number of combinations 
       #our function 'bc' is better.
       bcn = bc.bc0(n,m)
       # initialise the matrix of all m-combinations 
       P = np.zeros((bcn,m),precision);
       np1 = n + 1;  # do once here n+1 (needed in the internal loop)
       toRow = np1 - m 
       # set the first n+1-m rows of the last column
       temp=np.arange(m,n+1)
       temp=np.array([temp])
       temp=temp.T

       P[0:toRow , m-1] = temp[:,0]
       #print(P)
       
       for i in range(m-1,0,-1):                    # external loop over colums
          
          s1 = toRow
          s2 = toRow
          # set the first n+1-m rows block of rows of colum i
          P[0:toRow , i-1] = i
          ip1 = i + 1 # do once here i+1 (needed in the internal loop)
          for j in range(i+1, i+n-m+1):                 # internal loop
             s1 = int(s1*(np1+i-j-m)/(np1-j))
             fromRow = int(toRow + 1)
             toRow = int(fromRow + s1 - 1)
             P[fromRow-1:toRow , ip1-1:m] = P[s2-s1:s2 , ip1-1:m]
             P[fromRow-1:toRow , i-1] = j
          
       # find the true P if the vector of elements in v is not the default 1:n
       num=np.arange(n)
       P0=np.zeros(len(P))
       if not all(v==num):
           P0 = v[P-1]
       #print(P0)
       
    return P0
       
