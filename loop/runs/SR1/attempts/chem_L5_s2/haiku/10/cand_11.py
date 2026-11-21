import numpy as np
def equation(t, A, params):
    return params[0] + params[1]*t + params[2]*np.sqrt(t) + params[3]*np.log(t) + params[4]*A**params[5]*np.exp(-params[6]*t)
