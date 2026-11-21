import numpy as np
def equation(t, x, v, params):
    k, p, c, q, F, w, ph, b = params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7]
    return -k*np.abs(x)**p - c*np.abs(v)**q + F*np.cos(w*t+ph) + b
