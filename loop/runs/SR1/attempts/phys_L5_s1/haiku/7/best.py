import numpy as np
def equation(t, x, v, params):
    return params[0] + params[1]*np.sqrt(np.abs(v)) + params[2]*t + params[3]*np.exp(params[4]*t) + params[5]*t*t + params[6]*np.sin(params[7]*t)