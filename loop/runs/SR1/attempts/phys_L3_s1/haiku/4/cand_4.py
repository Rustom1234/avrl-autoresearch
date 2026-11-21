import numpy as np
def equation(t, x, v, params):
    w = params[0]
    return (params[1] + params[2]*x + params[3]*x**2 + params[4]*x**3 + params[5]*v + params[6]*v**2
            + params[7]*np.cos(w*t) + params[8]*np.sin(w*t))
