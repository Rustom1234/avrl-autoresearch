import numpy as np
def equation(t, x, v, params):
    F0, w, ph, d0, c, p, k1, k3, q = params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8]
    return F0*np.cos(w*t+ph) + d0 - c*np.sign(v)*np.abs(v)**p - k1*x - k3*x**3 - q*v*np.cos(w*t)
