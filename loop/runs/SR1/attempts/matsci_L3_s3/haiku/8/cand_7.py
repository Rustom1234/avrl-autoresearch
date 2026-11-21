import numpy as np
def equation(eps, T, params):
    a,b,c,d,e,f,g = params[0],params[1],params[2],params[3],params[4],params[5],params[6]
    return np.exp(a + b*T + c*eps + d*T*eps + e*np.log(eps) + f*T*np.log(eps) + g*eps**2)
