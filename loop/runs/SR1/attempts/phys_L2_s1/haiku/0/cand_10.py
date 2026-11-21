import numpy as np
def equation(t, x, v, params):
    w = params[7]
    return (params[0] + params[1]*x + params[2]*x**2 + params[3]*x**3 + params[4]*v + params[5]*v**2 + params[6]*v**3
            + params[8]*x*v + params[9]*x**2*v + params[10]*np.cos(w*t) + params[11]*np.sin(w*t))