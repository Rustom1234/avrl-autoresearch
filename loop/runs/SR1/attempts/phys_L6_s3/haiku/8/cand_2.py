import numpy as np
def equation(t, x, v, params):
    k, k3, c, F, w, ph, fc, b = params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7]
    return -k*x - k3*x**3 - c*v - fc*np.sign(v) + F*np.cos(w*t+ph) + b
