from scipy import special as sp
from scipy import stats as st
import math

def TBeff0(eff, p, *args):
    varargin = args
    nargin = 2 + len(varargin)
    eps=1e-12
    if (nargin<=2 or varargin[0] !=1):
        # LOCATION EFFICIENCY
        c= 2
        # c = starting point of the iteration
        # Remark: the more refined approximation for the starting value of
        # sqrt(chi2inv(eff,p))+2; does not seem to be necessary in the case of
        # location efficiency
    
        # step = width of the dichotomic search (it decreases by half at each
        # iteration).
        step=30
           
        # Convergence condition is E(\rho) = \rho(c) bdp
        #  where \rho(c) for TBW is c^2/6
        empeff=10
        p4=(p+4)*(p+2)
        p6=(p+6)*(p+4)*(p+2)
        p8=(p+8)*(p+6)*(p+4)*(p+2)
    
        # bet  = \int_{-c}^c  \psi'(x) d \Phi(x)
        # alph = \int_{-c}^c  \psi^2(x) d \Phi(x)
        # empeff = bet^2/alph = 1 / [var (robust estimator of location)]
    
        while abs(empeff-eff)> eps:
        
            cs=c**2/2
        
            bet= p4*sp.gammainc((p+4)/2, cs)/((c**4))\
                -2*(p+2)*sp.gammainc((p+2)/2, cs)/(c**2)+\
                + sp.gammainc(p/2, cs)
        
            alph= p8*sp.gammainc((p+10)/2, cs)/(c**8)-4*p6*sp.gammainc((p+8)/2, cs)/(c**6)+\
                6*p4*sp.gammainc((p+6)/2, cs)/(c**4)-2*(p+2)*sp.gammainc((p+4)/2, cs)/cs+\
                sp.gammainc((p+2)/2, cs)
            empeff=(bet**2)/alph
        
            step=step/2
            if empeff<eff:
                c=c+step
            elif empeff>eff:
                c=max(c-step,0.1)
            #c=math.sqrt(st.chi2.ppf(eff,p))+7
    else:
        # SHAPE EFFICIENCY
        if (nargin<=3 or varargin[1] !=1):
            # approxsheff 0 ==> use exact formulae
            approxsheff=0
        else:
            # approxsheff 1 ==> use Tyler approximation
            approxsheff=1
        # constant for second Tukey Biweight rho-function for MM, for fixed shape-efficiency
        # c = starting point of the iteration
        c=math.sqrt(st.chi2.ppf(eff,p))+7
        # step = width of the dichotomic search (it decreases by half at each
        # iteration).
        if (eff<0.92 and approxsheff==0):
            step=5
        else:
            step=15
        varrobestsc=10
        p4=(p+4)
        p6=p4*(p+6)
        p8=p6*(p+8)
        p10=p8*(p+10)
        # alphsc = E[ \psi^2(x) x^2] /{(v(v+2)]^2}
        # betsc =  E [ \psi'(x) x^2+(v+1)  \psi^2(x) x ]/[v(v+2)]
        # res = [var (robust estimator of scale)] = alphsc/(betsc^2)

        while abs(1-eff*varrobestsc)> eps:
            cs=c**2/2
            alphsc = sp.gammainc((p+4)/2,cs) \
                -4*p4*sp.gammainc((p+6)/2,cs)/(c**2)+\
                +6*p6*sp.gammainc((p+8)/2,cs)/(c**4)+\
                -4*p8*sp.gammainc((p+10)/2,cs)/(c**6)\
                +p10*sp.gammainc((p+12)/2,cs)/(c**8)
        
            betsc= sp.gammainc((p+2)/2,cs) \
                -2*p4*sp.gammainc((p+4)/2,cs)/(c**2)+\
                +p6*sp.gammainc((p+6)/2,cs)/(c**4)
        
            varrobestsc=alphsc/(betsc**2)
        
            if (p>1 and approxsheff==0):
            
                Erho2=p*(p+2)*sp.gammainc(0.5*(p+4),cs)/4-0.5*p*(p+2)*p4*sp.gammainc(0.5*(p+6),cs)/(c**2)\
                    +(5/12)*p*(p+2)*p6*sp.gammainc(0.5*(p+8),cs)/(c**4)-\
                    -(1/6)*p*(p+2)*p8*sp.gammainc(0.5*(p+10),cs)/(c**6)-\
                    +(1/36)*p*(p+2)*p10*sp.gammainc(0.5*(p+12),cs)/(c**8)
                
                Erho= p*sp.gammainc(0.5*(p+2),cs)/2-(p*(p+2))*0.5*sp.gammainc(0.5*(p+4),cs)/(c**2)+\
                    +p*(p+2)*(p+4)*sp.gammainc(0.5*(p+6),cs)/(6*(c**4))+ (cs*(1-sp.gammainc(p/2,cs))  )

                Epsixx= p*sp.gammainc(0.5*(p+2),cs)-2*(p*(p+2))*sp.gammainc(0.5*(p+4),cs)/(c**2)+\
                    +p*(p+2)*(p+4)*sp.gammainc(0.5*(p+6),cs)/(c**4)

                k3=(Erho2-Erho**2)/(Epsixx**2)

                varrobestsc=((p-1)/p)*varrobestsc+2*k3

        
            # disp(1-eff*varrobestsc)
        
            step=step/2
            if (1-eff*varrobestsc)<0:
                c=c+step
            elif (1-eff*varrobestsc)>0:
                c=max(c-step,0.1)
            
            #  disp([c 1-eff*res])
        

    ceff=c

    return ceff


        
        
    
