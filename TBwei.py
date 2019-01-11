
def TBwei0(u,c):
    
    ## beginning of code

    w = (1 - (u/c)**2)**2

    # The following instruction is unnecessary
    # however it is the proper expression for the weights
    # if we start with the normalized \rho (\infty)=1
    # w = w .* (c^2/6);

    w[ abs(u/c) > 1 ]= 0

    
    return w
