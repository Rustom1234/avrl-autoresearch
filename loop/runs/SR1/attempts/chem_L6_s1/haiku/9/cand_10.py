import numpy as np
def equation(t, A, params):
    a,b,c,d,e = params[0],params[1],params[2],params[3],params[4]
    return (a + b*np.exp(c*t))*A - d*t + e*A**2
