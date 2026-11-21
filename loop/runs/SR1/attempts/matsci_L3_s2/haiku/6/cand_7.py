import numpy as np
def equation(eps, T, params):
    a,b,c,d,e,f,g=params[0],params[1],params[2],params[3],params[4],params[5],params[6]
    return a + b*T + c*np.exp(-d*eps) + e*np.log(eps) + f*eps + g*eps*T