


def TBrho0(u,c):
    ## Beginning of code


    w = (abs(u)<=c)
    rhoTB = (u**2/(2)*(1-(u**2/(c**2))+(u**4/(3*c**4))))*w +(1-w)*(c**2/6)

    return rhoTB
