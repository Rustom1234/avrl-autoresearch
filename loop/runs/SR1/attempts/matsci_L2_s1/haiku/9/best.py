import numpy as np
def equation(eps, T, params):
    return -(params[0] + params[1]*T**params[2]) + (params[3] + params[4]*T**params[5]) * np.log(1 + params[6]*eps) + params[7]*np.exp(params[8]*T)*eps
