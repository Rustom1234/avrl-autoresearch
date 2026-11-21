import numpy as np
def equation(eps, T, params):
    return (params[0] + params[1]*T + params[2]*T**2 + params[3]*np.exp(-T)
            + (params[4] + params[5]*T + params[6]*T**2)*eps
            + (params[7] + params[8]*T)*eps**2 + params[9]*eps**2*np.exp(-T)*0)