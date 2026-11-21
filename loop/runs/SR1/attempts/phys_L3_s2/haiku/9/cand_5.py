import numpy as np
def equation(t, x, v, params):
    w=params[8]
    return params[0] + params[1]*x + params[2]*v + params[3]*x**2 + params[4]*x**3 + params[5]*v*x + params[6]*v**2 + params[7]*np.cos(w*t) + params[9]*np.sin(w*t)