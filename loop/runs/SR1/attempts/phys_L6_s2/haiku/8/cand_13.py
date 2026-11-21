import numpy as np
def equation(t, x, v, params):
    b,a1,k,pp,c,q,d,e = params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7]
    return b + a1*t + k*np.abs(x)**pp + c*np.abs(v)**q + d*t*x + e*t*t
