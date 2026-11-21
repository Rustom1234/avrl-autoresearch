import numpy as np
def equation(eps, T, params):
    a,b,c,p,d,e=params[0],params[1],params[2],params[3],params[4],params[5]
    return a + b*np.exp(-c*eps*T**p) + d*np.log(eps)*T**e