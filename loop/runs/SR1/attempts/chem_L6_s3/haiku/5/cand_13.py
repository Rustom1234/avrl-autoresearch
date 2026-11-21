import numpy as np
def equation(t, A, params):
    return params[0]*(1-np.exp(-params[1]*t))*np.exp(-params[2]*t) - params[3]*A**params[4]*t**params[5]/(1+params[7]*A) + params[6]
