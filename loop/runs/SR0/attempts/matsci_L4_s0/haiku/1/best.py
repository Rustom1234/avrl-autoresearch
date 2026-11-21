import numpy as np
def equation(eps, T, params):
    return params[0]*T**params[1]*(1-np.exp(-params[2]*eps)) + params[3]*T**params[4]*np.sqrt(eps) + params[5]*T**params[6]*np.exp(-params[7]*eps) + params[8]*eps*T**params[9]
