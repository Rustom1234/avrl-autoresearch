import numpy as np
def equation(t, x, v, params):
    w=params[0]
    return params[1]*np.cos(w*t)+params[2]*np.sin(w*t) - params[3]*x - params[4]*v - params[5]*x**2 - params[6]*x**3 - params[7]*v*np.abs(v) + params[8]*v*np.exp(-x) + params[9]
