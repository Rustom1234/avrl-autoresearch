import numpy as np
def equation(eps, T, params):
    return params[0] + params[1]*(1-np.exp(-params[2]*T)) + params[3]*eps**params[4]*np.exp(-params[5]*eps) + params[6]*(1-np.exp(-params[2]*T))*eps**params[4]*np.exp(-params[5]*eps)
