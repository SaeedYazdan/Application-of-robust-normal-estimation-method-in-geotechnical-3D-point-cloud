import ksiint as ks
from scipy import stats as st

def Tbsb0(c,p):
    y1 = ks.ksiint0(c,1,p)*3/c-ks.ksiint0(c,2,p)*3/(c**3)+ks.ksiint0(c,3,p)/(c**5)
    y2 = c*(1-st.chi2.cdf(c**2,p))
    res = y1+y2
    return res
