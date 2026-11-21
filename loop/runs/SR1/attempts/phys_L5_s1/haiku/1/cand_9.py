import numpy as np
def equation(t, x, v, params):
    return params[0] + params[1]*np.sin(params[2]*t+params[3]) + params[4]*v + params[5]*v**2 + params[6]*v**3 + params[7]*x + params[8]*x**3 + params[9]*t
