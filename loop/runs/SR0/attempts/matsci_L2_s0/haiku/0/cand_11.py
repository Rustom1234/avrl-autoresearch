import numpy as np
def equation(eps, T, params):
    return params[0] + params[1]*np.log(T) + params[2]/T + params[3]*T + params[4]*eps + params[5]*eps**2*T**params[6] + params[7]*T**2