import numpy as np
def equation(eps, T, params):
    a,b,c,n,d,e,f,g,h,k=params
    return -(a*np.exp(-b*T) + c*eps**n*np.exp(-d*T) + e*eps + f*eps*T + g*eps**2*np.exp(-h*T) + k*eps**(n)*T)