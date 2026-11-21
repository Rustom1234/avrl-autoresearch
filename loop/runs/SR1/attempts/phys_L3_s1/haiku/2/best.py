import numpy as np
def equation(t, x, v, params):
    F0, w, ph, c, k, b, d, G, ph2, e = params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9]
    return F0*np.cos(w*t+ph) + G*v*np.cos(w*t+ph2) - c*v - k*x - b*x**3 + d - e*np.abs(v)*v
