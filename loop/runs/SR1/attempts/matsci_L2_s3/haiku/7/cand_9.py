import numpy as np
def equation(eps, T, params):
    a,b,c,d,p,q,r = params[0],params[1],params[2],params[3],params[4],params[5],params[6]
    return a + b*T**p + (c + d*T**q)*np.abs(eps)**r