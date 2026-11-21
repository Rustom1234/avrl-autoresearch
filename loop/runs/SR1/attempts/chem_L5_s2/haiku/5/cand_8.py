import numpy as np
def equation(t, A, params):
    return params[0] + params[1]*np.log(t) + params[2]/t + params[3]*t + params[4]*A/t**params[5] + params[6]*A**params[7]
