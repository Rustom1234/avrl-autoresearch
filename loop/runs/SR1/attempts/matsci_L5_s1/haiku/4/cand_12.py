import numpy as np
def equation(eps, T, params):
    return (params[0] + params[1]*T + params[2]*T**2)*np.exp(-params[3]*eps) + params[4] + params[5]*eps + params[6]*T + params[7]*eps*T
