import numpy as np
def equation(t, x, v, params):
    b,F,w,ph,k,p1,c,d,ph2,c2,q = params
    return b + F*np.cos(w*t+ph) - k*np.abs(x)**p1 - (c+d*np.cos(w*t+ph2))*np.abs(v)**q - c2*np.abs(v)**q*0
