import numpy as np
def equation(t, x, v, params):
    F, w, ph, g0, g1, k0, k1, k2, b, c = params
    return F*np.cos(w*t+ph) + g0 + g1*t - (k0 + k1*t)*x + k2*x**2 - b*x**3 - c*v
