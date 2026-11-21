import numpy as np
def equation(eps, T, params):
    a,b,c,d,p,q,e,r = params
    return a + b*np.log(eps) + c/eps + d*eps**p*T**q + e*np.exp(-r*eps)
