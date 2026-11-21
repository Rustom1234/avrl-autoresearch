import numpy as np
def equation(eps, T, params):
    L = np.log(T)
    u = eps*T**params[6]
    return params[0] + params[1]*L + params[2]*u + params[3]*u**2 + params[4]*u**3 + params[5]*u**4 + params[7]*L**2*u + params[8]*L*u**2 + params[9]*L**2