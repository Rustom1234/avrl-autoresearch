import numpy as np
def equation(eps, T, params):
    return params[0] + params[1]*eps**2 + params[2]*T + params[3]*T**2 + params[4]*T**3 + params[5]*T**4 + params[6]*eps**2*T + params[7]*eps*T**2 + params[8]*eps**4 + params[9]*np.exp(-T)