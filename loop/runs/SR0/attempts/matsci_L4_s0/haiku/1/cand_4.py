import numpy as np
def equation(eps, T, params):
    return params[0]*T**params[1]*np.abs(eps-params[2])**params[3]*np.sign(eps-params[2]) + params[4]*T + params[5]
