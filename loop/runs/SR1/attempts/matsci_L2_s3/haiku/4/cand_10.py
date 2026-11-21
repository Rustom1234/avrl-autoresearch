import numpy as np
def equation(eps, T, params):
    a,b,c,d = params[0],params[1],params[2],params[3]
    return (a + b*np.log(T))*(1 + c*eps**2) + d*T**2
