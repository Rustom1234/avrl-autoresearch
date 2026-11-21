import numpy as np
def equation(eps, T, params):
    return params[0] + params[1]*np.log(T) + params[2]*eps + params[3]*np.exp(-params[4]*eps*T**params[5]) + params[6]*eps*np.exp(-params[7]*eps/T**params[8])
