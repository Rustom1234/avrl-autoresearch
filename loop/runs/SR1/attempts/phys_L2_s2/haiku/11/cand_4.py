import numpy as np
def equation(t, x, v, params):
    return params[0] + params[1]*x + params[2]*x**2 + params[3]*x**3 + params[4]*v + params[5]*x*v + params[6]/v + params[7]*np.sin(params[8]*t) + params[9]*np.cos(params[8]*t)