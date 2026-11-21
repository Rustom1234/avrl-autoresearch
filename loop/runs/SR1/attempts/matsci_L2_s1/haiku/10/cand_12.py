import numpy as np
def equation(eps, T, params):
    a,b,c,d,e,f,g,h,k,m = params
    return a + b*eps + c*np.exp(-d*T) + e*T**3 + f*eps*T**2 + g/T + h*eps/T + k*eps**2 + m*np.exp(-eps)