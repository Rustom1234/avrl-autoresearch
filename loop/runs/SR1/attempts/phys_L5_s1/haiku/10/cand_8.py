import numpy as np
def equation(t, x, v, params):
    F, w, ph, c1, k1, k3, c2, k2, c0 = params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8]
    return c0 + F*np.cos(w*t+ph) + c1*v + k1*x + k3*x**3 + c2*v*np.abs(v) + k2*x**2
