import numpy as np
def equation(t, x, v, params):
    w=params[0]
    return (params[1] + params[2]*v + params[3]*x + params[4]*v**2 + params[5]*x**2 + params[6]*x**3
            + (params[7] + params[8]*v)*np.cos(w*t) + (params[9] + params[10]*v)*np.sin(w*t))