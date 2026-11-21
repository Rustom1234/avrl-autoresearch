import numpy as np
def equation(t, x, v, params):
    w=params[8]
    return (params[0] + params[1]*x + params[2]*v + params[3]*x*v + params[4]*x**2 + params[5]*x**3
            + params[6]*np.cos(w*t) + params[7]*np.sin(w*t) + params[9]*v**3)
