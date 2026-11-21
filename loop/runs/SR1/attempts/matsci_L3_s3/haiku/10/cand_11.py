import numpy as np
def equation(eps, T, params):
    return params[0]*eps**params[1]*T**params[2] + params[3]*T + params[4]*np.log(eps)*T**params[5] + params[6]*eps**params[7]*T**params[8] + params[9]
