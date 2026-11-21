import numpy as np
def equation(t, x, v, params):
    a0,k,c,F,G,w,ph,d = params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7]
    return a0 + k*np.sqrt(x) + c/v + (F + G/v)*np.cos(w*t+ph) + d*x*np.cos(w*t+ph)
