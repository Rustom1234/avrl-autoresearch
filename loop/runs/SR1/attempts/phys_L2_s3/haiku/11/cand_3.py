import numpy as np
def equation(t, x, v, params):
    F, w, ph, c, k, b = params[0], params[1], params[2], params[3], params[4], params[5]
    return F*np.cos(w*t+ph) - c*v - k*x - b*x**3 + params[6]
