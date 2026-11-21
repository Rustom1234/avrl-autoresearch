import numpy as np
def equation(t, x, v, params):
    return params[0] + params[1]*x + params[2]*v + params[3]*x**3 + params[4]*x**2 + params[5]*np.cos(params[6]*t) + params[7]*np.sin(params[6]*t) + params[8]*v**2
