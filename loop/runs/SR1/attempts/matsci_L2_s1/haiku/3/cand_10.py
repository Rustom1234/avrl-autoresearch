import numpy as np
def equation(eps, T, params):
    a,b,c,d,e,f,g,h,p,q=params
    return a + b*T + c*np.exp(-d*T) + (e + f*T)*eps**p + g*eps*np.exp(-h*T) + q*eps**2
