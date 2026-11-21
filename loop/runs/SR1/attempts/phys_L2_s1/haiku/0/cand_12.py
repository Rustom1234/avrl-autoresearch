import numpy as np
def equation(t, x, v, params):
    w = 3.9225
    return (params[0] + params[1]*x + params[2]*x**2 + params[3]*x**3 + params[4]*v
            + params[5]*v**2 + params[6]*v**3 + params[7]*x*v
            + params[8]*np.cos(w*t) + params[9]*np.sin(w*t))