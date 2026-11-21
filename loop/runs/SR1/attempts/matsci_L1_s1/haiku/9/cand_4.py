import numpy as np
def equation(eps, T, params):
    a,b,c,d,e=params[0],params[1],params[2],params[3],params[4]
    return a*T**b*np.sin(c*eps+d)*np.exp(-e*eps)
