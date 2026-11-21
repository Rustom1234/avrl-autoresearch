import numpy as np
def equation(eps, T, params):
    return params[0]*np.log(eps) + params[1]*eps**params[2]*(params[3]+params[4]*T) + params[5]*T + params[6] + params[7]*T**2*np.log(eps)