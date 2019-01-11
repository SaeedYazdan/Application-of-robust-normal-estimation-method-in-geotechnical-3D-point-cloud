from scipy import stats as st
from scipy.special import gamma


def ksiint0(c,s,p):
    res = (2**s)*gamma(s+p/2)*st.gamma.cdf(c**2/2,s+p/2)/gamma(p/2)
    return res
