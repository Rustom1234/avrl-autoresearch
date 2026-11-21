import numpy as np
def equation(t, x, v, params):
    w=params[0]
    return params[1]*np.cos(w*t+params[2]) + params[3]*x + params[4]*v + params[5]*x**3 + params[6]*v*np.abs(v) + params[7]*x**2 + params[8]*v**2 + params[9]*x*v
