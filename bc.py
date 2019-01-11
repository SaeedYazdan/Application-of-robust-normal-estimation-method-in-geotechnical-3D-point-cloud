import math

def bc0(n,k):
    
    if k> n/2:
        k=n-k
    return round(math.factorial(n)/(math.factorial(k)*math.factorial(n-k)))
