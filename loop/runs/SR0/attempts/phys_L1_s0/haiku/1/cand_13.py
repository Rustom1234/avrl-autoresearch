import numpy as np
def equation(t, x, v, params):
    w = params[9]
    return (params[0] + params[1]*v + params[2]*v**2 + params[3]*v**3
            + params[4]*x + params[5]*x**2 + params[6]*x**3 + params[7]*x*v
            + params[8]*np.cos(w*t))
