import numpy as np
def equation(t, x, v, params):
    k1,k2,k3,c0,b,F0,w,ph = params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7]
    return -k1*x - k2*x**2 - k3*x**3 + (c0*np.exp(-b*t))*v*0 + F0*np.cos(w*t+ph) + params[8]*np.exp(-b*t)*v + params[9]*t*v
