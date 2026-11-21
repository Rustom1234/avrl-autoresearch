import numpy as np
def equation(t, x, v, params):
    w,A,phi,b,c,k,k3,d,e,f = params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7],params[8],params[9]
    return A*np.cos(w*t+phi) + b - c*v - k*x - k3*x**3 - d*v*np.abs(v) - e*v**3 + f*x*v