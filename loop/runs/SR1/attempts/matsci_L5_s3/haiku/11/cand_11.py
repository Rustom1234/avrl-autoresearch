import numpy as np
def equation(eps, T, params):
    return params[0] + params[1]*T + params[2]*np.log(T) + params[3]/T + params[4]*eps**params[5] + params[6]*eps
