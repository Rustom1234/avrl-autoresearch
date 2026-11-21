import numpy as np
def equation(t, x, v, params):
    F0,w,ph,a,b,c,d,e,f,g = params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7],params[8],params[9]
    return a + b*t + F0*np.cos(w*t+ph) + c*x + d*v + e*x**3 + f*v*np.abs(v) + g*x*v