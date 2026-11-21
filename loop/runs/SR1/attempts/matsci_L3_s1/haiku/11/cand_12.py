import numpy as np
def equation(eps, T, params):
    a,b,n,k,c,d,m = params[0],params[1],params[2],params[3],params[4],params[5],params[6]
    return a + b*eps**n*np.exp(-k*eps*T) + c*eps + d*T + m*np.log(T)