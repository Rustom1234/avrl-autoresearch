import numpy as np
def equation(eps, T, params):
    a,b,n,c,m,k,p = params[0],params[1],params[2],params[3],params[4],params[5],params[6]
    return (a + b*eps**n)*(1 - c*T**m) + k*eps*np.exp(-p*eps*T)