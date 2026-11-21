import numpy as np
def equation(eps, T, params):
    s = np.sqrt(1 + params[4]*eps) - 1
    return params[0]*eps + params[1]*eps*T + params[2]*T**3 + params[3]*s + params[5]*T*s
