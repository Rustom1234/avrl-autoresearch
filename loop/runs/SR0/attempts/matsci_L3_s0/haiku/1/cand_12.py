import numpy as np
def equation(eps, T, params):
    a,b,c,d,e,f,g = params[0],params[1],params[2],params[3],params[4],params[5],params[6]
    return a*T**b*(1-np.exp(-c*eps))*np.exp(-d*eps*T) + e*T**f*eps**g
