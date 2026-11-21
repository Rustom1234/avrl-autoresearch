import numpy as np
def equation(t, x, v, params):
    w = params[6]
    return params[0]*x*np.exp(-params[1]*x) + params[2]*x + params[3]*v + params[4]*np.cos(w*t) + params[5]*np.sin(w*t) + params[7]*x*x*np.exp(-params[8]*x)