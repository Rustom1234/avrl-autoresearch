import numpy as np
def equation(eps, T, params):
    a,b,c,p,q,d,e,f,g=params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7],params[8]
    return a + b*(T-c)**2 + (d+e*(T-c)**2)*eps*np.exp(-p*eps) + f*eps + g*eps**2