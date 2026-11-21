import numpy as np
def equation(eps, T, params):
    a,b,n,c,d=params[0],params[1],params[2],params[3],params[4]
    return -(a + b*eps**n) * (1 - np.exp(-c*T**d))
