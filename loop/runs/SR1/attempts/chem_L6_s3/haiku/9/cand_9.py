import numpy as np
def equation(t, A, params):
    a,b,p,c,m,q = params[0],params[1],params[2],params[3],params[4],params[5]
    return a*t**p*np.exp(-b*t) - c*A**m*t**q