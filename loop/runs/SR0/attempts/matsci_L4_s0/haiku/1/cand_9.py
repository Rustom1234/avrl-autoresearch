import numpy as np
def equation(eps, T, params):
    return params[0]*(1-np.exp(-params[1]*eps)) + (params[2]+params[3]*T)*eps + params[4]*T + params[5]*T*(1-np.exp(-params[1]*eps)) + params[6]
