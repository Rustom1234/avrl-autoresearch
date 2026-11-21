import numpy as np
def equation(t, A, params):
    a,b,c,d,e,f = params[0],params[1],params[2],params[3],params[4],params[5]
    return a*A**2 + b*A + c*np.exp(d*t) + e*A*np.exp(f*t)