import numpy as np
def equation(eps, T, params):
    A = params[0] + params[1]*T + params[2]*T**2 + params[3]*T**3
    return A*(1 - params[4]*np.sqrt(eps) - params[5]*eps) + params[6]*np.sqrt(eps)*(params[7]+params[8]*T+params[9]*T**2)
