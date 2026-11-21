import numpy as np
def equation(t, x, v, params):
    k, k3, c, c2, F, w, ph, b, G = params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8]
    return -k*x - k3*x**3 - c*v - c2*v*np.abs(v) + F*np.cos(w*t+ph) + b + G/v
