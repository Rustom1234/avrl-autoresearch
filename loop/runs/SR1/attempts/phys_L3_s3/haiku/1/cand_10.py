import numpy as np
def equation(t, x, v, params):
    F0, w, ph, c1, c2, k1, k3, g = params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7]
    return F0*np.cos(w*t+ph) - c1*v - c2*v*np.abs(v) - k1*x - k3*x**3 - g*x*v**2 + params[8]
