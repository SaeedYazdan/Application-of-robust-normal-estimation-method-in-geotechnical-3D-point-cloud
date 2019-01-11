import numpy as np
import math

def oneW(k,n):
    #print("k is:"+ str(k)+" and n is:"+str(n))
    if 4*k > n:
            # If the sample is a reasonable fraction of all combinations,
            # just randomize the whole population and take the first nsel.
            # Note that function shuffling (see the 'utilities' folder)
            # randomises the combinations without calling function sort.
            rp=np.arange(n)
            random.shuffle(rp)
            y = rp[1:k]
            
    else:
            # If the desired sample is small compared to all combinations,
            # it may be more convenient to repeatedly sample with
            # replacement until there are nsel unique values.
            mindiff = 0
            while mindiff == 0:
                # sample w/replacement
                y=np.random.randint(n, size=k)
                y.sort()
                print(y)
                a=np.diff(y)
                print(a)
                #y = randi(n, 1 , k)
                mindiff = min(a)
    return y


def twoW(k,n):
    # Systematic sampling method, Cochran (1977), third edition, Sampling
    # Techniques, Wiley.
    stepk=math.floor(n/k);
    startk=np.random.randint(n);
    y=np.arange(startk,startk+stepk*k,stepk)
    z=y[y>n]
    g=np.argwhere(y>n)
    y[g]=y[g]-n

    return y

def threeW(k,n):
    print("this part has not been developed completely")
    # A Linear Congruential Generator (LCG) represents one of the
    # oldest and best-known pseudorandom number generator algorithms

    # Triple of Lehmer
    m = 2**31 - 1
    a = 16807
    c = 0
        
    y = np.zeros(k)
    t = np.zeros(k)
    y[0] =  np.random.randint(n)
    #print(y)
    for i in range(k):
        if i>0:
            y[i] = (a * y[i-1] + c)% m
        t[i]=math.ceil(y[i]*n/m)

    return t



def randsampleFS0(n,k,method=1):
    options = {
           1 : oneW,
           2 : twoW,
           3 : threeW
    }
    y=options[method](k,n)

    return y
