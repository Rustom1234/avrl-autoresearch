import numpy as np
def equation(eps, T, params):
    a,b,c,p,q,d,f,g = params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7]
    return (a + b/T + c*T) * eps**p*np.exp(-q*eps) + d + f*eps + g*np.log(T)
