import numpy as np
def equation(eps, T, params):
    a,b,c,d,e,f,g=params[0],params[1],params[2],params[3],params[4],params[5],params[6]
    return a*eps + b*eps*T**2 + c*T**2*np.exp(-d*eps) + e*eps**2 + f*T**2*eps**2 + g*np.exp(-d*eps)*eps
