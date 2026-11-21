import numpy as np
def equation(t, x, v, params):
    w=params[0]
    return params[1]*np.cos(w*t)+params[2]*np.sin(w*t) - params[3]*v - params[4]*x - params[5]*x**3 - params[6]*v*np.abs(v) - params[7] - params[8]*x**2 - params[9]*x*v
