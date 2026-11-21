import numpy as np
def equation(t, A, params):
    a,b,c,d,e,f = params[0],params[1],params[2],params[3],params[4],params[5]
    return a*np.exp(-b*t**2)*A - c*t + d*np.exp(e*t)/A + f*A**2
