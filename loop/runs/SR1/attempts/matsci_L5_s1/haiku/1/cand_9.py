import numpy as np
def equation(eps, T, params):
    return params[0]+params[1]*eps+params[2]*T+params[3]*np.sin(params[4]*eps+params[5]*T+params[6])+params[7]*eps*eps
