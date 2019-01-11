

def fw0(u,c):
    
    # weight function = psi(u)/u
    tmp = (1 - (u/c)**2)**2
    tmp = tmp * (c**2/6)
    tmp[ abs(u/c) > 1 ]= 0
    return tmp
