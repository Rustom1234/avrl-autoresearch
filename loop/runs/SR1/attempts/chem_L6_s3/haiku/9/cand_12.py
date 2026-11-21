import numpy as np
def equation(t, A, params):
    a,b,d,c,m,q = params[0],params[1],params[2],params[3],params[4],params[5]
    return a*(np.exp(-b*t)-np.exp(-d*t)) - c*A**m*t**q