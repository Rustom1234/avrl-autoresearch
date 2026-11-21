import numpy as np
def equation(eps, T, params):
    return params[0] + params[1]*eps**1.5 + params[2]*np.sqrt(T) + params[3]*T**params[4] + params[5]*T*np.sqrt(T) + params[6]*np.log(T)