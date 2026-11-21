import numpy as np
def equation(t, x, v, params):
    return params[0] + params[1]*t + params[2]*t**2 + params[3]*v + params[4]*v**2 + params[5]*x + params[6]*np.sin(params[7]*x+params[8])
