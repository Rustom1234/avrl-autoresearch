import numpy as np
def equation(eps, T, params):
    a,b,c,d,e,f,g,h,k,m,n=params
    return a + b*T + c*np.exp(-d*T) + eps*(e + f*T + g*np.exp(-h*T)) + k*eps**2*np.exp(-m*T) + n*eps**3
