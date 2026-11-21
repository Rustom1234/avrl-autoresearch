import numpy as np
def equation(t, x, v, params):
    return params[0] + params[1]*x + params[2]*x**2 + params[3]*x**3 + params[4]*v + params[5]*v**2 + params[6]*np.exp(-params[7]*t)
