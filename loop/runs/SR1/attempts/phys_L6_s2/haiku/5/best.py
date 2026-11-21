import numpy as np
def equation(t, x, v, params):
    a,b,c,d,p,q,e = params[0],params[1],params[2],params[3],params[4],params[5],params[6]
    return a + b*np.abs(v)**p + c*t**q + d*t/x + e/x