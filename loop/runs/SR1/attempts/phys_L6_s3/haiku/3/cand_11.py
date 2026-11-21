import numpy as np
def equation(t, x, v, params):
    w=params[0]
    return params[1]*np.cos(w*t+params[2]) + params[3]*x + params[4]*x**3 + params[5]*v + params[6]*v*np.abs(v) + params[7] + params[8]*x*v*v + params[9]*v**3
