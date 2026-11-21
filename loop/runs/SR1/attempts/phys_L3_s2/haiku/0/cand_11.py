import numpy as np
def equation(t, x, v, params):
    w = params[10]
    return (params[0] + params[1]*x + params[2]*x**2 + params[3]*x**3
            + params[4]*v + params[5]*x*v + params[6]*x*x*v + params[7]*v*v + params[8]*v**3
            + params[9]*np.cos(w*t) + 0.0*np.sin(w*t))
