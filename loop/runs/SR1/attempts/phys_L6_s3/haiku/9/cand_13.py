import numpy as np
def equation(t, x, v, params):
    a0,k,c,F,G,w,ph,p,q = params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7],params[8]
    return a0 + k*x**p + c*v**(-q) + (F + G/v)*np.cos(w*t+ph)
