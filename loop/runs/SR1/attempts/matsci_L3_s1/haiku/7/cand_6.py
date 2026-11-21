import numpy as np
def equation(eps, T, params):
    a,b,c,p,q,d,e=params[0],params[1],params[2],params[3],params[4],params[5],params[6]
    return d + (a + b*(T-c)**2)*eps**p*np.exp(-q*eps) + e*eps