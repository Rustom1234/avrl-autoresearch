import numpy as np
def equation(eps, T, params):
    a,b,c,d,n,e,f=params[0],params[1],params[2],params[3],params[4],params[5],params[6]
    return -(a + b/np.sqrt(T)) - (c + d*T)*eps**n - e*eps*np.sqrt(T) - f*T
