import numpy as np
def equation(t, x, v, params):
    w=params[6]
    return (params[0] + params[1]*x + params[2]*x**2 + params[3]*v + params[4]*v**2
            + params[5]*np.cos(w*t) + params[7]*np.sin(w*t) + params[8]*x*np.cos(w*t) + params[9]*v*np.cos(w*t))
