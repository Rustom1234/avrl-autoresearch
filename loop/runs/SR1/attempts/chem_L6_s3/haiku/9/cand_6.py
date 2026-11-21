import numpy as np
def equation(t, A, params):
    a,b,c,m,p = params[0],params[1],params[2],params[3],params[4]
    return a*np.exp(-b*t) - c*A**m*t**p