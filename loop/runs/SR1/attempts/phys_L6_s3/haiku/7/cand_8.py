import numpy as np
def equation(t, x, v, params):
    F,w,ph,c,p,k,q,d=params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7]
    return F*np.cos(w*t+ph) - c*np.abs(v)**p - k*np.abs(x)**q - d
