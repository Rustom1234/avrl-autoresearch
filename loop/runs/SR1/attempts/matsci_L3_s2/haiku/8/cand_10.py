import numpy as np
def equation(eps, T, params):
    g = 1-np.exp(-params[0]*T)
    return params[1] + params[2]*g + (params[3]+params[4]*T)*eps**params[5]*np.exp(-(params[6]+params[7]*T)*eps) + params[8]*g*eps**params[5]*np.exp(-params[6]*eps)
