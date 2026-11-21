import numpy as np
def equation(eps, T, params):
    a,b,c,d,e,f = params[0],params[1],params[2],params[3],params[4],params[5]
    return (a + b*T) + (c + d*T)*(1-np.exp(-e*eps)) + f*eps
