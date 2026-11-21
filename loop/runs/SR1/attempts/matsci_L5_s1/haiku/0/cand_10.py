import numpy as np
def equation(eps, T, params):
    s=np.sqrt(eps)
    return params[0]*np.exp(-params[1]*T) + params[2] + params[3]*T + (params[4] + params[5]*T + params[9]*T**2)*s + params[6]*eps + params[7]*eps*T + params[8]*np.exp(-params[1]*T)*s
