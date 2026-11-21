import numpy as np
def equation(t, x, v, params):
    F0, w, phi, c, k, k3, b0 = params[0], params[1], params[2], params[3], params[4], params[5], params[6]
    return F0*np.cos(w*t+phi) - c*v - k*x - k3*x**3 + b0
