import numpy as np
def equation(t, x, v, params):
    F, w, ph, c, k, k3, c0 = params[0], params[1], params[2], params[3], params[4], params[5], params[6]
    return F*np.cos(w*t+ph) - c*v - k*x - k3*x**3 + c0