import numpy as np
def equation(eps, T, params):
    return params[0] + params[1]*np.log(eps) + params[2]*eps**params[3]*np.exp(params[4]*T) + params[5]*T + params[6]*np.log(eps)*T
