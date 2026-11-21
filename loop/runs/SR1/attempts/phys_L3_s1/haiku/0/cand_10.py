import numpy as np
def equation(t, x, v, params):
    F0, w, ph, b, d0, c1, c2, c3, k1 = params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8]
    return F0*np.exp(-b*t)*np.cos(w*t+ph) + d0 + c1*v + c2*v*np.abs(v) + c3*v**3 + k1*x
