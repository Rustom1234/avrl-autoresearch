import numpy as np
def equation(eps, T, params):
    return params[0] + params[1]*np.log(T) + params[2]/T + params[3]*T + params[4]*T**2 + params[5]*eps + params[6]*eps**2*T**2 + params[7]*eps**3 + params[8]*eps**2