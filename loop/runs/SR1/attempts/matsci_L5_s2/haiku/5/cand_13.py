import numpy as np
def equation(eps, T, params):
    a,b,c,d,f,g = params[0],params[1],params[2],params[3],params[4],params[5]
    return a*(1-np.exp(-b*eps**c)) + d*eps**f + g*np.log(1+eps)*(1+0*T)
