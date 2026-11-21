import numpy as np
def equation(eps, T, params):
    a,b,c,d,p=params[0],params[1],params[2],params[3],params[4]
    return (a + b*T**p)*(1 + c*np.exp(-d*eps))