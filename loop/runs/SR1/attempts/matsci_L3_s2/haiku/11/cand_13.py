import numpy as np
def equation(eps, T, params):
    a,b,c,d,p,g,k,q = params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7]
    return a + b*(1-np.exp(-c*T)) + d*eps**p*np.exp(-g*T) + k*eps**q
