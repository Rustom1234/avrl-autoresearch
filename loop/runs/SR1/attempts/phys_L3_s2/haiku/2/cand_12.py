import numpy as np
def equation(t, x, v, params):
    w=params[9]
    return (params[0] + params[1]*x + params[2]*v + params[3]*x*v + params[4]*x**2 + params[5]*x**3
            + params[6]*v**2 + params[7]*np.cos(w*t) + params[8]*np.sin(w*t)
            + params[10]*v**3 + params[11]*x*v**2 + params[12]/x + params[13]/v)
