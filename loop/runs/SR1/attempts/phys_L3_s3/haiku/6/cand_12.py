import numpy as np
def equation(t, x, v, params):
    w = params[9]
    return (params[0] + params[1]*x + params[2]*v + params[3]*x**3 + params[4]*v**3
            + params[5]*x*v + params[6]*x**2*v + params[7]*np.cos(w*t) + params[8]*np.sin(w*t))
