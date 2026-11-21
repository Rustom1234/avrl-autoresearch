import numpy as np
def equation(t, x, v, params):
    F0, w, ph, c, k1, k3, c2 = params[0], params[1], params[2], params[3], params[4], params[5], params[6]
    return F0*np.cos(w*t+ph) - c*v - k1*x - k3*x**3 - c2*v*np.abs(v)
