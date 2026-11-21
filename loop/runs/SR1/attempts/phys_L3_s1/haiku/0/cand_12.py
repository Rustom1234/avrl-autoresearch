import numpy as np
def equation(t, x, v, params):
    F0, w, ph, d0, c, h, ph2, k1, k3, q, s = params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[0]*0
    return F0*np.cos(w*t+ph) + d0 - c*v - h*v*np.cos(w*t+ph2) - k1*x - k3*x**3 - q*v**3
