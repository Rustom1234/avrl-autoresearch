import numpy as np
def equation(eps, T, params):
    return params[0] + params[1]*T + params[2]*eps + params[3]*eps*T + params[4]*eps**2 + params[5]*T**2 + params[6]*eps**2*T + params[7]*eps*T**2 + params[8]*eps**3 + params[9]*np.log(eps)
