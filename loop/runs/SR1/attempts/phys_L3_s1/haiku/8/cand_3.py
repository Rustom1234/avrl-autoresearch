import numpy as np
def equation(t, x, v, params):
    A,w,phi,c,k,b,d,e = params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7]
    return A*np.cos(w*t+phi) - c*v - k*x - b*x**3 - d*x**2 - e*v*np.abs(v)
