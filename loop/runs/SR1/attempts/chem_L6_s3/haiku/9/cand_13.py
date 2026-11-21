import numpy as np
def equation(t, A, params):
    a,b,p,c,m,q,e = params[0],params[1],params[2],params[3],params[4],params[5],params[6]
    return a*t**p*np.exp(-b*t) - c*A**m*t**q + e*A*t*np.exp(-b*t)