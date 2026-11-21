import numpy as np
def equation(t, x, v, params):
    a,b,c,d,e = params[0],params[1],params[2],params[3],params[4]
    return a + b/x + c*np.sqrt(np.abs(v)) + d*np.sqrt(t) + e*t/x