import numpy as np
def equation(t, x, v, params):
    w = params[0]
    return (params[1]*np.sin(w*t) + params[2]*np.cos(w*t) + params[3]
            + params[4]*x + params[5]*x**3 + params[6]*v + params[7]*v*np.abs(v) + params[8]*x*v + params[9]*v**3)
