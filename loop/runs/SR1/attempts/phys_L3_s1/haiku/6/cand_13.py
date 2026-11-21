import numpy as np
def equation(t, x, v, params):
    b,F,w,ph,k,k3,c,d,ph2,c2 = params
    return b + F*np.cos(w*t+ph) - k*x - k3*x**3 - (c+d*np.cos(w*t+ph2))*v - c2*v*np.abs(v) - 0.0*x*v